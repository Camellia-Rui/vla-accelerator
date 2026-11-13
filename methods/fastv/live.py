"""Process-local recording hooks for the original OpenVLA LIBERO evaluator."""
import hashlib
import json
from pathlib import Path
import time
from core.live_evaluation import atomic, publish


def state_hash(state):
    import numpy as np
    a=np.asarray(state,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape)).encode()+a.tobytes()).hexdigest()


def publish_modes(output, report):
    atomic(output/'live-episodes.json',report)
    records=list(report['records'])
    other='baseline' if report['protocol']['modes']==['fastv'] else 'fastv'
    sibling=output.parent/other/'live-episodes.json'
    if sibling.exists():
        peer=json.loads(sibling.read_text())
        if {k:v for k,v in peer['protocol'].items() if k!='modes'} != {k:v for k,v in report['protocol'].items() if k!='modes'}:
            raise ValueError('FastV paired protocols differ')
        records+=peer['records']
    combined=dict(protocol=dict(report['protocol'],modes=['baseline','fastv']),records=records)
    publish(output.parent,combined)
    return combined


class OpenVLARecorder:
    def __init__(self, output, mode, trials, seed):
        self.output=Path(output);self.mode=mode;self.row=None;self.error=None
        self.report=dict(protocol=dict(tasks=list(range(10)),trials=list(range(trials)),seed=seed,
                                      modes=[mode],action_chunk=1),records=[])
        self.restore=[]

    def finish_episode(self):
        if self.row is None:return
        row=self.row
        if self.error:raise RuntimeError(self.error)
        if not row['queries']:raise ValueError('Episode has no recorded policy queries')
        if not row['control_steps']:raise ValueError('Episode has no executed actions')
        row.update(complete=True,episode_seconds=self.last_step-self.started,
                   policy_ms=sum(q['ms'] for q in row['queries']))
        self.report['records'].append(row)
        self.row=None
        publish_modes(self.output,self.report)

    def install(self):
        import numpy as np
        import torch
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from experiments.robot import robot_utils
        suite=benchmark.get_benchmark_dict()['libero_spatial']()
        states={}
        for t in self.report['protocol']['tasks']:
            initial=suite.get_task_init_states(t)
            for i in self.report['protocol']['trials']:
                h=state_hash(initial[i])
                if h in states:raise ValueError('Ambiguous initial-state identity')
                states[h]=(t,i)
        original_state=OffScreenRenderEnv.set_init_state
        original_step=OffScreenRenderEnv.step
        original_action=robot_utils.get_action
        def set_state(env,state,*a,**kw):
            self.finish_episode()
            h=state_hash(state)
            if h not in states:raise ValueError('Unexpected initial state')
            t,i=states[h]
            self.row=dict(task_id=t,trial=i,mode=self.mode,seed=self.report['protocol']['seed'],
                          initial_state_hash=h,success=False,error=None,complete=False,queries=[],
                          executed_actions=[],control_steps=0)
            return original_state(env,state,*a,**kw)
        def action(*a,**kw):
            try:
                torch.cuda.synchronize();start=time.perf_counter()
                if self.row is None:raise ValueError('Query without initial state')
                if not self.row['queries']:self.started=start
                value=original_action(*a,**kw)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                values=np.asarray(value)
                if values.shape!=(7,) or not np.isfinite(values).all():raise ValueError('Invalid OpenVLA action')
                self.row['queries'].append(dict(ms=elapsed,actions=[values.tolist()]))
                return value
            except Exception as exc:
                self.error=str(exc);raise
        def step(env,act,*a,**kw):
            try:
                value=original_step(env,act,*a,**kw)
                if self.row is not None and self.row['queries']:
                    self.last_step=time.perf_counter()
                    self.row['executed_actions'].append(np.asarray(act).tolist())
                    self.row['control_steps']+=1
                    self.row['success']=bool(value[2])
                return value
            except Exception as exc:
                self.error=str(exc);raise
        for obj,name,replacement in ((OffScreenRenderEnv,'set_init_state',set_state),
                                     (OffScreenRenderEnv,'step',step),(robot_utils,'get_action',action)):
            self.restore.append((obj,name,getattr(obj,name)));setattr(obj,name,replacement)
        return self

    def close(self, success):
        try:
            if success:
                self.finish_episode()
                publish(self.output,self.report,final=True)
            else:
                atomic(self.output/'assessment.json',dict(status='FAILED',quality_decision='INCONCLUSIVE',error=self.error or 'Runner failed'))
        finally:
            for obj,name,original in reversed(self.restore):setattr(obj,name,original)
