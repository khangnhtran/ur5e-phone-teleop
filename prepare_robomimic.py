"""Export successful human-phone data with official robomimic tools."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import h5py
import numpy as np
from check_dataset import check

def validate_source(source):
    source=Path(source).resolve()
    if any(p.lower() in ("validation","fixtures","synthetic") for p in source.parts):
        raise ValueError("Fixture/synthetic directories are not training sources.")
    with h5py.File(str(source),"r") as f:
        data=f["data"]
        teleop=json.loads(data.attrs.get("teleop_config","{}"))
        if teleop.get("validation_only") or "action_fixture" in teleop or teleop.get("data_origin") in ("synthetic","fixture","external_unverified"):
            raise ValueError("Fixture, synthetic or unverified external source: export refused.")
        meta=json.loads(data.attrs["env_args"])
        cfg=meta["env_kwargs"]
        robots=cfg.get("robots")
        robots=[robots] if isinstance(robots,str) else robots
        assert meta["env_name"]=="PickPlaceCan" and robots==["UR5e"]
        assert cfg["controller_configs"]["type"]=="OSC_POSE"
        assert float(cfg.get("control_freq",20))==20
        cfg["control_freq"]=20
        names=list(data)
    if not names:
        raise ValueError("No successful demonstrations saved yet.")
    return meta,names

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    p.add_argument("--output-dir")
    p.add_argument("--output-prefix",default="")
    p.add_argument("--val-ratio",type=float,default=.2)
    p.add_argument("--split-seed",type=int,default=0)
    a=p.parse_args()
    if (a.output_prefix and Path(a.output_prefix).name!=a.output_prefix) or not 0<a.val_ratio<1:
        p.error("invalid prefix or validation ratio")
    source=Path(a.dataset).resolve()
    metadata,names=validate_source(source)
    checksum=hashlib.sha256(source.read_bytes()).hexdigest()
    result=check(str(source),demo=names[0],mode="states")
    if not result["valid"]:
        raise SystemExit("Raw data check or state playback failed; no conversion.")
    output=Path(a.output_dir).resolve() if a.output_dir else source.parent
    prefix=a.output_prefix+"_" if a.output_prefix else ""
    converted=output/(prefix+"robomimic.hdf5")
    lowdim=output/(prefix+"low_dim.hdf5")
    if converted.exists() or lowdim.exists():
        raise SystemExit("Output already exists. Use a new --output-dir; no overwrite.")
    output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(str(source),str(converted))
    shim=Path(__file__).resolve().parent/"robomimic_native.py"
    def run(module,*options):
        subprocess.run([sys.executable,"-B",str(shim),module]+list(options),check=True)
    run("robomimic.scripts.conversion.convert_robosuite","--dataset",str(converted))
    with h5py.File(str(converted),"a") as f:
        f["data"].attrs["env_args"]=json.dumps(metadata)
        demo_names=np.array(sorted(f["data"]))
        np.random.RandomState(a.split_seed).shuffle(demo_names)
        nval=max(1,min(len(demo_names)-1,int(round(len(demo_names)*a.val_ratio)))) if len(demo_names)>1 else 0
        masks=f.require_group("mask")
        for key,values in (("valid",demo_names[:nval]),("train",demo_names[nval:])):
            if key in masks:
                del masks[key]
            masks.create_dataset(key,data=np.array([x.encode() for x in values],dtype="S64"))
    run("robomimic.scripts.dataset_states_to_obs","--dataset",str(converted),
        "--output_name",lowdim.name,"--done_mode","2","--compress")
    run("check_observations","--raw",str(source),"--low",str(lowdim),"--repair-terminal","--essential",
        "--report",str(output/(prefix+"data_check.json")))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=checksum:
        raise RuntimeError("Raw input changed unexpectedly.")
    report=dict(raw=str(source),raw_sha256=checksum,raw_check=result,
        converted=str(converted),low_dim=str(lowdim),raw_unchanged=True,
        data_origin="human_phone",split_by_demo=True,train_demos=len(demo_names)-nval,valid_demos=nval)
    (output/(prefix+"export.json")).write_text(json.dumps(report,indent=2))
    print("READY:",lowdim,flush=True)
if __name__=="__main__":
    main()
