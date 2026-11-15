"""iPhone UR5e/Can recording. B3 starts, success saves, Ctrl-C exports low_dim."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import numpy as np
import robosuite as suite
from basic_recording import BasicEpisode
CONTROL_FREQ=20
from demo_store import DemoStore,atomic_json
from phone_adapter import PhoneAdapter
from local_receiver import LocalReceiver
from top_down import TOP_DOWN_R,TopDownResetWrapper,base_env

ROOT=Path(__file__).resolve().parent

class PinEdges:
    def __init__(self,pin="B3",timeout=.25):
        self.timeout=timeout
        self.pin=pin
        self.last=None
        self.session=None
    def update(self,m,now):
        if m.get("session")!=self.session:
            self.session=m.get("session")
            self.last=None
        value=m.get("buttons",{}).get(self.pin)
        age=m.get("button_age_s",{}).get(self.pin,1e9)
        if value not in (0,1) or age is None or not np.isfinite(age) or age<0 or age+max(0,now-m.get("host_monotonic",now))>self.timeout:
            self.last=None
            return False
        edge=value==1 and self.last==0
        self.last=value
        return edge

def arguments():
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config",default=str(ROOT/"collection_config.json"))
    known,_=pre.parse_known_args()
    config=json.loads(Path(known.config).read_text())
    if config["control_freq"]!=CONTROL_FREQ:
        raise SystemExit("This collector is fixed at 20Hz")
    p=argparse.ArgumentParser(description=__doc__,parents=[pre])
    for key in ("ip","family","name","output_root"):
        p.add_argument("--"+key.replace("_","-"),default=config[key])
    for key in ("pos_scale","rot_scale","smooth_tau","max_speed","max_accel"):
        p.add_argument("--"+key.replace("_","-"),type=float,default=config[key])
    p.add_argument("--port",type=int,default=config["port"])
    p.add_argument("--phone-timeout-seconds",type=float,default=config["phone_timeout_seconds"])
    p.add_argument("--quality-timeout-seconds",type=float,default=config["quality_timeout_seconds"])
    p.add_argument("--mode",choices=["practice","record"],default=config["mode"])
    p.add_argument("--tcp-orientation",choices=["topdown","phone"],default=config["tcp_orientation"])
    p.add_argument("--orientation-control",choices=["phone","locked"],default=config.get("orientation_control","phone"))
    p.add_argument("--max-angular-speed",type=float,default=config.get("max_angular_speed",1.8))
    p.add_argument("--max-lag-m",type=float,default=config.get("max_lag_m",.04))
    p.add_argument("--max-lag-rad",type=float,default=config.get("max_lag_rad",.30))
    p.add_argument("--require-quality",action="store_true",default=config["require_quality"])
    p.add_argument("--headless",action="store_true")
    p.add_argument("--no-eef-axes",action="store_true",help="hide TCP dot")
    p.add_argument("--axes-length",type=float,default=.15,help="soft local Z guide length in metres")
    p.add_argument("--translation-only",action="store_true")
    p.add_argument("--external-receiver",action="store_true")
    p.add_argument("--receiver-python",default=str(ROOT/".venv-receiver/bin/python"),help="HEBI receiver Python, separate from simulation env")
    p.add_argument("--seconds",type=float,default=0,help="test runtime; 0 unlimited")
    p.add_argument("--seed",type=int)
    p.add_argument("--runtime-dir",default=str(ROOT),help=argparse.SUPPRESS)
    p.add_argument("--max-demos",type=int,default=config["max_demos"],help="accepted demo limit; 0 unlimited")
    p.add_argument("--gripper-button",default="B2",choices=["B2","B4","B5","B6","B7"])
    p.add_argument("--reset-button",default="B8",choices=["B4","B5","B6","B7","B8"])
    p.add_argument("--success-seconds",type=float,default=config["policy"]["success_seconds"])
    p.add_argument("--no-export",action="store_true",help="skip automatic conversion on exit")
    a=p.parse_args()
    a.receiver_python=str(Path(a.receiver_python).expanduser().resolve())
    output=Path(a.output_root).expanduser()
    a.output_root=str((output if output.is_absolute() else ROOT/output).resolve())
    if not 0<a.success_seconds<=5:
        p.error("invalid success hold duration")
    if not (.05<=a.phone_timeout_seconds<=2 and .05<=a.quality_timeout_seconds<=5):
        p.error("invalid timeout")
    if a.gripper_button==a.reset_button or a.max_demos<0:
        p.error("invalid buttons or demo limit")
    if not (.2<=a.max_angular_speed<=3 and .01<=a.max_lag_m<=.08 and .1<=a.max_lag_rad<=.5):
        p.error("invalid following limits")
    if not (0<a.pos_scale<=2 and 0<a.rot_scale<=2 and .01<=a.smooth_tau<=.5
            and .02<=a.max_speed<=.5 and .1<=a.max_accel<=3):
        p.error("trajectory parameters outside configured limits")
    return a

def make_driver(env,args):
    c=env.robots[0].controller
    c.update(force=True)
    d=PhoneAdapter(c,args.pos_scale,args.rot_scale,args.translation_only,not args.require_quality,
        args.gripper_button,args.reset_button,args.smooth_tau,args.max_speed,args.max_accel,
        orientation_lock=c.ee_ori_mat.copy() if args.orientation_control=="locked" else None,
        stale_seconds=args.phone_timeout_seconds,quality_stale_seconds=args.quality_timeout_seconds,
        max_angular_speed=args.max_angular_speed,max_lag_m=args.max_lag_m,max_lag_rad=args.max_lag_rad)
    centers=np.array([env.bin1_pos,env.bin2_pos])
    half=np.asarray(env.bin_size)[:2]/2
    d.lower=np.r_[np.minimum(np.min(centers[:,:2]-half-.08,axis=0),c.ee_pos[:2]-.1),
                  min(env.bin1_pos[2],env.bin2_pos[2])+.01]
    d.upper=np.r_[np.maximum(np.max(centers[:,:2]+half+.08,axis=0),c.ee_pos[:2]+.1),
                  max(c.ee_pos[2]+.2,env.bin1_pos[2]+.45)]
    if not args.headless and not args.no_eef_axes:
        from eef_axes import attach_axes
        attach_axes(env,args.axes_length)
    return d

def main():
    a=arguments()
    runtime=Path(a.runtime_dir).resolve()
    runtime.mkdir(parents=True,exist_ok=True)
    lock=(runtime/".collector.lock").open("a")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another collector is running; stop it before starting a new session.")
    receiver=LocalReceiver(a.port)
    stop=[False]
    def on_signal(*unused):
        stop[0]=True
    signal.signal(signal.SIGINT,on_signal)
    signal.signal(signal.SIGTERM,on_signal)
    pidfile=runtime/".collector.pid"
    pidfile.write_text(str(os.getpid()))
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
    session=Path(a.output_root).expanduser().resolve()/((a.mode+"_")+stamp)
    child=log=store=env=stats=None
    completed=0
    started=time.monotonic()
    next_print=started
    deadline=started
    packets=0
    m={}
    start_edges=PinEdges(timeout=a.phone_timeout_seconds)
    loss_latched=None
    retry_at=0.
    if a.seed is not None:
        np.random.seed(a.seed) # once per session, never force a fixed can position
    config=dict(env_name="PickPlaceCan",robots=["UR5e"],
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        control_freq=CONTROL_FREQ,use_camera_obs=False,ignore_done=True,reward_shaping=True)
    teleop=dict(vars(a))
    teleop["control_freq"]=CONTROL_FREQ
    teleop["simple_flow"]=True
    teleop["data_origin"]="external_unverified" if a.external_receiver else "human_phone"
    if not a.external_receiver:
        version_code="import json,sys,importlib.metadata as m; print(json.dumps(dict(python=sys.version,hebi_py=m.version(\"hebi-py\"),numpy=m.version(\"numpy\"))))"
        teleop["receiver_versions"]=json.loads(subprocess.check_output([a.receiver_python,"-I","-c",version_code],text=True))
    else:
        teleop["receiver_versions"]={"source":"external_localhost_UDP"}
    if a.tcp_orientation=="topdown":
        teleop["tcp_rotation_world"]=TOP_DOWN_R.tolist()
    try:
        raw_env=suite.make(**config,has_renderer=not a.headless,has_offscreen_renderer=False,render_camera="agentview")
        if a.mode=="record":
            store=DemoStore(raw_env,session,config,teleop)
            env=store.env
        else:
            env=TopDownResetWrapper(raw_env) if a.tcp_orientation=="topdown" else raw_env
            session.mkdir(parents=True,exist_ok=False)
            atomic_json(session/"session.json",dict(mode="practice",env=config,teleop=teleop))
        env.reset()
        d=make_driver(env,a)
        practice_steps=0
        practice_sim_start=float(env.sim.data.time)
        practice_wall_start=time.monotonic()
        def event(kind,**data):
            if store:
                store.event(kind,**data)
            else:
                with (session/"events.jsonl").open("a") as f:
                    f.write(json.dumps(dict(event=kind,**data))+"\n")
        def finish(reason,accepted=False):
            nonlocal stats,completed
            if stats is not None:
                summary=stats.summary(time.monotonic())
                store.finish(accepted,reason,summary,stats.samples)
                if accepted:
                    completed+=1
                print("END",json.dumps(dict(reason=reason,accepted=accepted,**summary)),flush=True)
                stats=None
        if not a.external_receiver:
            log=(session/"receiver.log").open("a",buffering=1)
            cmd=[a.receiver_python,"-I",str(ROOT/"phone_probe.py"),
                 "--lookup-address",a.ip,"--family",a.family,"--name",a.name,
                 "--discover-seconds","3","--udp-port",str(a.port),"--stale",str(a.phone_timeout_seconds)]
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        print("SESSION",session,flush=True)
        print("MODE",a.mode,"| B1 hold | B2 gripper | B3 reset+START recording | B8 CANCEL/reset",flush=True)
        event("session_start",mode=a.mode)
        while not stop[0] and (not a.seconds or time.monotonic()-started<a.seconds):
            now=time.monotonic()
            if child is not None and child.poll() is not None and now>=retry_at:
                event("receiver_retry",wall_elapsed=now-started)
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                retry_at=now+5.
            start=False
            for message in receiver.drain():
                packets+=1
                m=message
                timestamp=time.monotonic()
                d.ingest(message,timestamp)
                if d.is_healthy(timestamp):
                    start |= start_edges.update(message,timestamp)
            if d.reset_requested:
                finish("operator_cancel",False)
                env.reset()
                d=make_driver(env,a)
                receiver.drain()
                loss_latched=None
                practice_steps=0
                practice_sim_start=float(env.sim.data.time)
                practice_wall_start=time.monotonic()
                event("operator_reset")
            elif start and a.mode=="record" and stats is None:
                env=store.begin(time.monotonic()) # fresh reset, XML/state persisted before first recorded step
                d=make_driver(env,a)
                stats=BasicEpisode(env.sim.data.time,time.monotonic(),a.success_seconds)
                receiver.drain()
                loss_latched=None
                event("record_start_wait_release_B1")
            now=time.monotonic()
            # Apply timeout before gating; advance the trajectory only once per simulator step.
            healthy=d.is_healthy(now)
            can_step=healthy and d.armed
            status=("RECORDING" if stats else ("PRACTICE" if a.mode=="practice" else "PREVIEW_WAIT_B3"))
            d.check_timeout(now)
            action=np.r_[np.zeros(6),d.grip_drive]
            if not can_step:
                status="PAUSED_"+d.status
                if stats and d.last_fresh is not None and not healthy and loss_latched!=d.status:
                    event("tracking_pause",step=stats.steps,reason=d.last_error or d.status,
                          sim_time=float(env.sim.data.time),wall_elapsed=now-stats.wall_start)
                    loss_latched=d.status
                # No env.step: simulator and sample count remain frozen through outages.
            else:
                if loss_latched is not None:
                    event("tracking_resumed",step=stats.steps if stats else 0)
                    loss_latched=None
                c=env.robots[0].controller
                c.update(force=True)
                action=d.action(now,control_dt=1/CONTROL_FREQ)
                low,high=env.action_spec
                if not np.isfinite(action).all() or np.any(action<low-1e-8) or np.any(action>high+1e-8):
                    raise RuntimeError("action outside environment limits")
                if stats is not None:
                    env.step(action)
                    c.update(force=True)
                    accepted=stats.observe(env.sim.data.time,time.monotonic(),action,env._check_success())
                    if accepted:
                        finish("stable_task_success",True)
                        env.reset()
                        d=make_driver(env,a)
                        receiver.drain()
                        if a.max_demos and completed>=a.max_demos:
                            event("accepted_demo_limit",accepted=completed)
                            break
                elif a.mode=="practice":
                    base_env(env).step(action) # practice never records through data wrapper
                    practice_steps+=1
                    if not np.isclose(float(env.sim.data.time)-practice_sim_start,practice_steps/CONTROL_FREQ,atol=1e-7,rtol=0):
                        raise RuntimeError("practice simulator clock mismatch")
                # Record mode waits for B3 without stepping or accumulating preview samples.
            sim_seconds=stats.sim_end-stats.sim_start if stats else (float(env.sim.data.time)-practice_sim_start if a.mode=="practice" else 0.)
            wall_seconds=time.monotonic()-stats.wall_start if stats else (time.monotonic()-practice_wall_start if a.mode=="practice" else 0.)
            displayed_steps=stats.steps if stats else (practice_steps if a.mode=="practice" else 0)
            lines=[a.mode.upper()+" | "+status,
                   "steps=%d | sim=%.2fs | steps/20=%.2fs | wall=%.2fs"%
                   (displayed_steps,sim_seconds,displayed_steps/20,wall_seconds),
                   "saved=%d | hold=%d/%d | packets=%d"%(completed,stats.hold if stats else 0,max(1,int(np.ceil(a.success_seconds*20))),packets)]
            if m.get("ar_quality") is None:
                lines.append("AR tracking quality unavailable; pose timeout protection enabled")
            if not a.headless:
                base_env(env).viewer.hud_lines=lines
                env.render()
            if now>=next_print:
                print(json.dumps(dict(mode=a.mode,recording=stats is not None and can_step,status=status,
                    phone_status=d.status,phone_error=d.last_error,steps=displayed_steps,
                    sim_seconds=sim_seconds,steps_over_20=displayed_steps/20,
                    wall_seconds=wall_seconds,accepted=completed,packets=packets,
                    buttons=m.get("buttons",{}),gripper_closed_request=d.closed,
                    gripper_drive=float(action[-1]),quality="unknown_allowed" if m.get("ar_quality") is None else m.get("ar_quality"))),flush=True)
                next_print=now+1
            deadline+=1/CONTROL_FREQ
            delay=deadline-time.monotonic()
            if delay>0:
                time.sleep(delay)
            else:
                deadline=time.monotonic() # no burst of catch-up steps
    finally:
        if stats is not None and store is not None:
            store.finish(False,"shutdown",stats.summary(time.monotonic()),stats.samples)
        if env is not None:
            env.close()
        receiver.close()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=3)
        if log:
            log.close()
        if pidfile.exists() and pidfile.read_text().strip()==str(os.getpid()):
            pidfile.unlink()
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
        print("Stopped; all recorded attempts retained:",session,flush=True)
        if store is not None and store.count and not a.no_export:
            if a.external_receiver:
                print("External source not verified as human phone; automatic training export skipped.",flush=True)
            else:
                print("Exporting successful phone demonstrations...",flush=True)
                result=subprocess.run([sys.executable,"-B",str(ROOT/"prepare_robomimic.py"),str(session/"demo.hdf5")])
                if result.returncode:
                    print("Export failed. Raw demo.hdf5 is preserved; inspect the error and rerun prepare_robomimic.py.",flush=True)
                    raise SystemExit(result.returncode)

if __name__=="__main__":
    main()
