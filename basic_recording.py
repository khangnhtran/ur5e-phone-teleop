"""Only sample timing and the environment's original success predicate."""
import math
import numpy as np

class BasicEpisode:
    def __init__(self,sim_start,wall_start,success_seconds=.5):
        self.sim_start=float(sim_start)
        self.sim_end=float(sim_start)
        self.wall_start=wall_start
        self.steps=0
        self.hold=0
        self.required=max(1,int(math.ceil(success_seconds*20)))
        self.task_success=False
        self.samples=[]
    def observe(self,sim_time,wall_time,action,task_success):
        self.steps+=1
        self.sim_end=float(sim_time)
        if not np.isclose(self.sim_end-self.sim_start,self.steps/20,atol=1e-7,rtol=0):
            raise RuntimeError("simulator clock does not match 20Hz samples")
        self.task_success=bool(task_success)
        self.hold=self.hold+1 if self.task_success else 0
        self.samples.append([self.sim_end,wall_time-self.wall_start,int(self.task_success)])
        return self.hold>=self.required
    def summary(self,wall_time):
        return dict(steps=self.steps,sim_start=self.sim_start,sim_end=self.sim_end,
            sim_seconds=self.sim_end-self.sim_start,wall_seconds=wall_time-self.wall_start,
            control_freq=20,task_success=self.task_success,
            success_hold_steps=self.hold,required_hold_steps=self.required)
