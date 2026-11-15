"""Read-only structural audit. State playback and action playback are distinct."""
import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np

def config_from_data(data):
    if "env_args" in data.attrs:
        return json.loads(data.attrs["env_args"])["env_kwargs"]
    return json.loads(data.attrs["env_info"])

def validate_structure(path, strict=True):
    report=dict(dataset=str(Path(path).resolve()),valid=True,demos=[],errors=[],warnings=[])
    with h5py.File(path,"r") as f:
        data=f["data"]
        config=config_from_data(data)
        frequency=float(config.get("control_freq",20))
        report["control_freq"]=frequency
        bounds=json.loads(data.attrs.get("action_bounds","null"))
        if bounds is None:
            import robosuite as suite
            runtime=dict(config)
            runtime.update(has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False)
            env=suite.make(**runtime)
            try:
                bounds=[np.asarray(x).tolist() for x in env.action_spec]
            finally:
                env.close()
        low,high=np.asarray(bounds[0]),np.asarray(bounds[1])
        total=0
        for name in sorted(data,key=lambda s:int(s.split("_")[-1])):
            g=data[name]
            a,s=g["actions"][:],g["states"][:]
            n=len(a)
            item=dict(demo=name,steps=n,action_shape=list(a.shape),state_shape=list(s.shape),
                      accepted=bool(g.attrs.get("accepted",True)),
                      task_success=g.attrs.get("task_success",None),
                      needs_review=bool(g.attrs["needs_review"]) if "needs_review" in g.attrs else None)
            if item["task_success"] is not None:
                item["task_success"]=bool(item["task_success"])
            errors=[]
            if a.ndim!=2 or a.shape[1]!=len(low) or s.ndim!=2 or len(s)!=n or n==0:
                errors.append("shape")
            if not np.isfinite(a).all() or not np.isfinite(s).all():
                errors.append("NaN_or_Inf")
            violations=int(np.count_nonzero((a<low-1e-7)|(a>high+1e-7))) if a.ndim==2 and a.shape[1]==len(low) else -1
            item["action_limit_violations"]=violations
            if violations:
                errors.append("action_limits")
            if "model_file" not in g.attrs:
                errors.append("missing_XML")
            if "num_samples" in g.attrs and g.attrs["num_samples"]!=n:
                errors.append("num_samples")
            if len(s)>1 and not np.allclose(np.diff(s[:,0]),1/frequency,atol=1e-7,rtol=0):
                errors.append("simulator_time_step")
            terminal=g["terminal_state"][:] if "terminal_state" in g else None
            if terminal is not None:
                if terminal.shape!=s.shape[1:] or not np.isfinite(terminal).all():
                    errors.append("terminal_state")
                item["sim_seconds"]=float(terminal[0]-s[0,0])
                if not np.isclose(item["sim_seconds"],n/frequency,atol=1e-7,rtol=0):
                    errors.append("sim_time_vs_steps")
            else:
                item["sim_seconds"]=n/frequency
                report["warnings"].append(name+": legacy file has no recorded terminal_state")
            item["steps_over_control_freq"]=n/frequency
            if "step_metrics" in g:
                metrics=g["step_metrics"][:]
                if metrics.shape!=(n,3) or not np.isfinite(metrics).all():
                    errors.append("step_metrics")
                else:
                    expected=np.r_[s[1:,0],terminal[0]] if terminal is not None else None
                    if expected is not None and not np.allclose(metrics[:,0],expected,atol=1e-7,rtol=0):
                        errors.append("metrics_time_alignment")
            if "episode_metadata" in g.attrs:
                item["metadata"]=json.loads(g.attrs["episode_metadata"])
                required=item["metadata"].get("required_hold_steps",10)
                if "step_metrics" in g and (n<required or not np.all(g["step_metrics"][-required:,2]==1)):
                    errors.append("success_hold_evidence")
            item["errors"]=errors
            report["errors"].extend(name+":"+x for x in errors)
            report["demos"].append(item)
            total+=n
        if "total" in data.attrs and data.attrs["total"]!=total:
            report["errors"].append("total_samples")
        if "mask" in f:
            names=set(data)
            for key,ds in f["mask"].items():
                keys={v.decode() if isinstance(v,bytes) else str(v) for v in ds[:]}
                if not keys<=names:
                    report["errors"].append("unknown_mask_demo:"+key)
    report["demo_count"]=len(report["demos"])
    report["total_samples"]=total
    report["valid"]=not report["errors"]
    if strict and not report["valid"]:
        raise ValueError(json.dumps(report,indent=2))
    return report

def replay(path, name, mode="states", render=False, tolerance=1e-5):
    import robosuite as suite
    with h5py.File(path,"r") as f:
        data=f["data"]
        cfg=dict(config_from_data(data))
        group=data[name]
        states=group["states"][:]
        actions=group["actions"][:]
        terminal=group["terminal_state"][:] if "terminal_state" in group else None
        xml=group.attrs["model_file"]
        expected=bool(group.attrs.get("task_success",group.attrs.get("successful",True)))
        success_flags=group["step_metrics"][:,2] if "step_metrics" in group else None
        meta=json.loads(group.attrs.get("episode_metadata","{}"))
        teleop=json.loads(data.attrs.get("teleop_config","{}"))
    cfg.update(has_renderer=render,has_offscreen_renderer=False,render_camera="agentview")
    env=suite.make(**cfg)
    error=0.
    first=None
    successes=[]
    try:
        env.reset()
        env.reset_from_xml_string(env.edit_model_xml(xml))
        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        if mode=="states":
            sequence=list(states)+(list([terminal]) if terminal is not None else [])
            for s in sequence:
                env.sim.set_state_from_flattened(s)
                env.sim.forward()
                successes.append(bool(env._check_success()))
                error=max(error,float(np.max(np.abs(env.sim.get_state().flatten()-s))))
                if render:
                    env.render()
                    time.sleep(.05)
            success_flags_match=success_flags is None or np.array_equal(np.asarray(successes[1:],dtype=int),success_flags.astype(int))
            passed=error<=tolerance and successes[-1]==expected and success_flags_match
        else:
            c=env.robots[0].controller
            c.update(force=True)
            if "controller_initial_joint" in meta:
                c.update_initial_joints(np.array(meta["controller_initial_joint"]))
                c.goal_ori=np.array(meta["controller_initial_goal_ori"])
            elif teleop.get("tcp_orientation")=="topdown":
                from top_down import sync_controller
                sync_controller(env)
            for i,action in enumerate(actions):
                env.step(action)
                expected_state=states[i+1] if i+1<len(states) else terminal
                if expected_state is not None:
                    e=float(np.max(np.abs(env.sim.get_state().flatten()-expected_state)))
                    error=max(error,e)
                    if e>tolerance and first is None:
                        first=dict(step=i,error=e)
                if render:
                    env.render()
                    time.sleep(.05)
            successes=[bool(env._check_success())]
            success_flags_match=success_flags is None or np.array_equal(np.asarray(successes[1:],dtype=int),success_flags.astype(int))
            passed=error<=tolerance and successes[-1]==expected and success_flags_match
        result=dict(demo=name,mode=mode,steps=len(actions),passed=passed,
                    max_state_error=error,first_divergence=first,final_task_success=successes[-1])
        if mode=="states":
            result["recorded_success_matches_states"]=bool(success_flags_match)
        return result
    finally:
        env.close()

def check(path,demo=None,render=False,tolerance=1e-5,mode="states",strict=True):
    result=validate_structure(path,strict=strict)
    result["playback"]=[]
    if mode!="none":
        names=[demo] if demo else [x["demo"] for x in result["demos"]]
        for name in names:
            result["playback"].append(replay(path,name,mode,render,tolerance))
    result["valid"]=result["valid"] and all(x["passed"] for x in result["playback"])
    brief={k:v for k,v in result.items() if k!="demos"}
    brief["demos"]=[{k:v for k,v in d.items() if k!="metadata"} for d in result["demos"]]
    print(json.dumps(brief,indent=2),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    p.add_argument("--demo")
    p.add_argument("--render",action="store_true")
    p.add_argument("--replay",choices=["states","actions","none"],default="states")
    p.add_argument("--tolerance",type=float,default=1e-5)
    p.add_argument("--report")
    a=p.parse_args()
    r=check(a.dataset,a.demo,a.render,a.tolerance,a.replay,strict=False)
    if a.report:
        Path(a.report).write_text(json.dumps(r,indent=2))
    raise SystemExit(0 if r["valid"] else 1)
