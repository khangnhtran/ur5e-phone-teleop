"""Validate each observation against its recorded simulator state."""
import argparse
import json
import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

def process(raw_path,low_path,repair_terminal=False,essential=False):
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs":{"low_dim":[],"rgb":[]}})
    with h5py.File(raw_path,"r") as raw, h5py.File(low_path,"r+" if repair_terminal else "r") as low:
        metadata=json.loads(raw["data"].attrs["env_args"])
        actual=json.loads(low["data"].attrs["env_args"])
        cfg=actual["env_kwargs"]
        assert actual["env_name"]=="PickPlaceCan" and cfg["robots"]==["UR5e"]
        assert cfg["controller_configs"]["type"]=="OSC_POSE" and cfg["control_freq"]==20
        env=EnvUtils.create_env_for_data_processing(metadata,camera_names=[],camera_height=84,
                    camera_width=84,reward_shaping=False)
        checked=0
        transitions=0
        try:
            for name,source in raw["data"].items():
                g=low["data"][name]
                states=source["states"][:]
                n=len(states)
                assert all(k in g for k in ("actions","states","obs","next_obs","rewards","dones"))
                assert g.attrs["num_samples"]==n and g.attrs["model_file"]==source.attrs["model_file"]
                assert g["rewards"].shape==(n,) and g["dones"].shape==(n,)
                transitions+=n
                terminal=source["terminal_state"][:]
                np.testing.assert_array_equal(g["actions"][:],source["actions"][:])
                np.testing.assert_array_equal(g["states"][:],states)
                env.reset_to({"model":source.attrs["model_file"],"states":states[0]})
                # Official extractor steps its last action. Use the RECORDED terminal
                # state instead on the derived copy; never change actions or raw data.
                if repair_terminal:
                    for key in ("accepted","task_success","needs_review","episode_metadata","end_reason","source_episode"):
                        if key in source.attrs:
                            g.attrs[key]=source.attrs[key]
                    obs=env.reset_to({"states":terminal})
                    for key in g["next_obs"]:
                        g["next_obs"][key][-1]=obs[key]
                    g["rewards"][-1]=env.get_reward()
                    g["dones"][-1]=1
                indices=[0] if essential else range(len(states))
                for i in indices:
                    state=states[i]
                    obs=env.reset_to({"states":state})
                    for key in g["obs"]:
                        np.testing.assert_allclose(g["obs"][key][i],obs[key],rtol=1e-6,atol=1e-7)
                    checked+=1
                final=env.reset_to({"states":terminal})
                assert np.isfinite(g["rewards"][:]).all() and np.isfinite(g["dones"][:]).all()
                assert set(np.unique(g["dones"][:]))<={0,1} and g["dones"][-1]==1
                for key in g["obs"]:
                    values=g["obs"][key][:]
                    next_values=g["next_obs"][key][:]
                    assert len(values)==n and next_values.shape==values.shape
                    assert np.isfinite(values).all() and np.isfinite(next_values).all()
                    np.testing.assert_allclose(next_values[:-1],values[1:],rtol=1e-6,atol=1e-7)
                    np.testing.assert_allclose(next_values[-1],final[key],rtol=1e-6,atol=1e-7)
            train={x.decode() for x in low["mask/train"][:]}
            valid={x.decode() for x in low["mask/valid"][:]}
            assert not train&valid and train|valid==set(low["data"])
            result=dict(demos=len(raw["data"]),states_checked=checked,transitions_checked=transitions,
                        essential_checks=essential,
                        observation_alignment=True,terminal_from_recorded_state=True,
                        train_demos=len(train),validation_demos=len(valid),split_by_whole_demo=True,
                        observation_features={k:list(v.shape[1:]) for k,v in g["obs"].items()})
            print(json.dumps(result,indent=2),flush=True)
            return result
        finally:
            env.env.close()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--raw",required=True)
    p.add_argument("--low",required=True)
    p.add_argument("--repair-terminal",action="store_true")
    p.add_argument("--essential",action="store_true")
    p.add_argument("--report",required=True)
    a=p.parse_args()
    result=process(a.raw,a.low,a.repair_terminal,a.essential)
    from pathlib import Path
    Path(a.report).write_text(json.dumps(result,indent=2))
