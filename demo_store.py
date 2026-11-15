"""Retain every attempt; export accepted demos with official states/actions alignment."""
import datetime
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
import hashlib
import h5py
import numpy as np
import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper
from robomimic.envs.env_base import EnvType

def atomic_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+".pending")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(str(temp),str(path))

def read_raw(path):
    states,actions=[],[]
    successful=False
    for chunk in sorted(Path(path).glob("state_*.npz")):
        with np.load(str(chunk),allow_pickle=True) as d:
            states.extend(d["states"])
            actions.extend(x["actions"] for x in d["action_infos"])
            successful |= bool(d["successful"])
    states,actions=np.asarray(states),np.asarray(actions)
    if len(states)!=len(actions)+1:
        raise ValueError("raw alignment must be N+1 states / N actions")
    return states,actions,successful

class DemoStore:
    def __init__(self, base_env, directory, env_config, teleop_config):
        self.directory=Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=False)
        self.raw=self.directory/"raw"
        self.raw.mkdir()
        self.env_config=env_config
        self.teleop_config=teleop_config
        self.versions={p:importlib.metadata.version(p) for p in
                       ("numpy","mujoco","robosuite","robomimic","torch")}
        self.versions["python"]=sys.version
        self.source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path(__file__).resolve().parent.glob("*.py")}
        wrapper=DataCollectionWrapper
        if teleop_config.get("tcp_orientation")=="topdown":
            from top_down import TopDownDataCollectionWrapper
            wrapper=TopDownDataCollectionWrapper
        self.env=wrapper(base_env,str(self.raw),collect_freq=1,flush_freq=100)
        self.manifest=[]
        self.active=None
        self.count=0
        atomic_json(self.directory/"session.json",dict(env=env_config,teleop=teleop_config,
            versions=self.versions,source_sha256=self.source_hashes,format_version=2))
        atomic_json(self.directory/"episodes.json",self.manifest)
        self.export()  # valid empty file; no invented demos

    def event(self,event,**data):
        with (self.directory/"events.jsonl").open("a") as f:
            f.write(json.dumps(dict(event=event,host_monotonic=time.monotonic(),utc=datetime.datetime.utcnow().isoformat()+"Z",**data),allow_nan=False)+"\n")
            f.flush()

    def begin(self,wall_time):
        if self.active is not None:
            raise RuntimeError("finish the current attempt before reset")
        self.env.reset()
        # Native first-interaction hook: persist XML and initial state BEFORE step.
        self.env._on_first_interaction()
        path=Path(self.env.ep_directory)
        np.save(str(path/"initial_state.npy"),self.env._current_task_instance_state)
        self.active=dict(episode=path.name,raw_path=str(path.relative_to(self.directory)),
                         accepted=False,task_success=False,reason="in_progress",
                         wall_started=float(wall_time),sim_started=float(self.env.sim.data.time))
        robot=self.env.robots[0]
        self.active["gripper"]=type(robot.gripper).__name__
        self.active["controller_initial_joint"]=robot.controller.initial_joint.tolist()
        self.active["controller_initial_goal_ori"]=robot.controller.goal_ori.tolist()
        obj=self.env.objects[self.env.object_id]
        self.active["can_initial_position"]=self.env.sim.data.body_xpos[self.env.obj_body_id[obj.name]].tolist()
        atomic_json(path/"config.json",dict(env=self.env_config,teleop=self.teleop_config,initial=self.active))
        self.env._flush()  # raw initial state survives even an empty canceled attempt
        self.event("begin",**self.active)
        return self.env

    def finish(self, stable_success=False, reason="reset", summary=None, samples=None):
        if self.active is None:
            return False
        self.env._flush()
        self.env.has_interaction=False
        path=self.directory/self.active["raw_path"]
        states,actions,native_ever=read_raw(path)
        native_final=bool(self.env._check_success())
        summary=summary or dict(steps=len(actions),sim_seconds=float(states[-1,0]-states[0,0]),
                               needs_review=True,review_reasons=["missing_metrics"])
        summary=summary or {}
        hold_ok=summary.get("success_hold_steps",0)>=summary.get("required_hold_steps",10)
        accepted=bool(stable_success and hold_ok and native_final and native_ever and len(actions)>0)
        if stable_success and not accepted:
            raise ValueError("cannot accept an empty or non-successful attempt")
        record=dict(self.active)
        record.update(summary)
        record.update(accepted=accepted,task_success=native_final,native_ever_success=native_ever,reason=reason)
        record["accepted"]=accepted
        record["task_success"]=native_final
        record["steps"]=len(actions)
        atomic_json(path/"episode.json",record)
        if samples is not None:
            np.save(str(path/"step_metrics.npy"),np.asarray(samples,dtype=float))
        self.manifest.append(record)
        atomic_json(self.directory/"episodes.json",self.manifest)
        self.event("finish",**record)
        self.active=None
        self.count=sum(r["accepted"] for r in self.manifest)
        self.export()
        return accepted

    def export(self):
        temp=self.directory/"demo.pending.hdf5"
        with h5py.File(str(temp),"w") as f:
            data=f.create_group("data")
            minimal={k:self.env_config[k] for k in ("env_name","robots","controller_configs")}
            data.attrs.update(dict(env="PickPlaceCan",env_info=json.dumps(minimal),
                repository_version=suite.__version__,date=datetime.date.today().isoformat(),
                time=datetime.datetime.now().strftime("%H:%M:%S"),format_version=2,
                teleop_config=json.dumps(self.teleop_config),software_versions=json.dumps(self.versions),
                env_args=json.dumps(dict(type=EnvType.ROBOSUITE_TYPE,env_name="PickPlaceCan",
                      env_version=suite.__version__,env_kwargs=self.env_config))))
            data.attrs["action_bounds"]=json.dumps([np.asarray(x).tolist() for x in self.env.action_spec])
            data.attrs["control_freq"]=self.env.control_freq
            data.attrs["robots"]=json.dumps(self.env_config["robots"])
            data.attrs["gripper"]=type(self.env.robots[0].gripper).__name__
            data.attrs["seed"]=json.dumps(self.teleop_config.get("seed"))
            total=0
            exported=0
            simple=self.teleop_config.get("simple_flow",False)
            clean,review=[],[]
            for record in self.manifest:
                if not record["accepted"]:
                    continue
                path=self.directory/record["raw_path"]
                states,actions,successful=read_raw(path)
                assert successful and np.isfinite(states).all() and np.isfinite(actions).all()
                exported+=1
                name="demo_%d"%exported
                g=data.create_group(name)
                # Official aggregator removes LAST state, not initial state.
                g.create_dataset("states",data=states[:-1],compression="gzip")
                g.create_dataset("actions",data=actions,compression="gzip")
                g.create_dataset("terminal_state",data=states[-1])
                g.create_dataset("sim_times",data=states[:-1,0])
                metrics=path/"step_metrics.npy"
                if metrics.exists():
                    g.create_dataset("step_metrics",data=np.load(str(metrics)),compression="gzip")
                g.attrs.update(dict(model_file=(path/"model.xml").read_text(),num_samples=len(actions),
                    successful=True,task_success=record["task_success"],accepted=True,
                    needs_review=record.get("needs_review",True),source_episode=record["episode"],
                    episode_metadata=json.dumps(record),end_reason=record["reason"]))
                if simple:
                    del g.attrs["needs_review"]
                total+=len(actions)
                (review if record.get("needs_review",True) else clean).append(name.encode())
            data.attrs["total"]=total
            if not simple:
                masks=f.create_group("mask")
                masks.create_dataset("needs_review",data=np.array(review,dtype="S64"))
                masks.create_dataset("clean",data=np.array(clean,dtype="S64"))
        os.replace(str(temp),str(self.directory/"demo.hdf5"))
