"""R7P训练后30例固定开发诊断；独立启动，不阻塞四路训练首发。"""
import argparse,os,sys,time
from pathlib import Path
from r7p_ops import *
def run(config_path):
 cfg=read(config_path);root=Path(cfg['workroot']).resolve();runtime=read(root/'runtime.json');verify(runtime['frozen_inputs'])
 state=read(root/'controller-state.json')
 if state['status']!='training_completed':raise ValueError('All four L48 workers must complete')
 sys.path.insert(0,cfg['evaluation_code'])
 from evaluation_r7p_runner import make_evaluate_case
 from r7p_evaluation import run_endpoint
 env=dict(os.environ,**runtime['child_env'])
 for key in runtime.get('child_env_remove',[]):env.pop(key,None)
 os.environ.clear();os.environ.update(env)
 deadline=epoch(runtime['lease_expires_at_utc'])-60
 if time.time()>=deadline:raise TimeoutError('Lease deadline')
 callback=make_evaluate_case(root,runtime['python'],cfg['evaluator'],cfg['study_root'],cfg['data_root'],deadline,device='cuda:0',startup_lock=root/'evaluation-init.lock',bootstrap_env=cfg.get('bootstrap_env'))
 parent=read(root/'parent-selection.json')['selected'];arms=[r['training_arm'] for r in state['arms']]
 return run_endpoint(root,parent,arms,callback)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
