"""最小CPU契约，不冒称真实GPU奖励已验证；运行时首步另做真实manager断言。"""
import ast,copy,hashlib,json,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from r7p_training import *
from r7p_train import write
class Fake:
 def __init__(self):
  self.config=NS(collection=NS(num_envs=128,steps_per_iteration=8,warmup_iterations=16,capacity_per_env=512),algorithm=NS(critic_updates_per_iteration=128,batch_size=256,n_steps=5,actor_update_every=2,alpha_update_every=1,target_update_every=1,auxiliary_updates_per_iteration=128))
  self.counters=NS(**{k:0 for k in ('iterations','env_steps','transitions','critic_updates','actor_updates','alpha_updates','target_updates','learning_skipped_iterations','auxiliary_updates','auxiliary_skipped')})
 def collect_steps(self,n):self.counters.env_steps+=n;self.counters.transitions+=128*n;return {}
 def learn_from_replay(self,n):
  for k,v in dict(critic_updates=n,alpha_updates=n,target_updates=n,actor_updates=n//2).items():setattr(self.counters,k,getattr(self.counters,k)+v)
  return {}
 def auxiliary_update(self):self.counters.auxiliary_updates+=1;return {}
class Contracts(unittest.TestCase):
 def test_exact_plan(self):validate_plan(json.loads(Path(__file__).with_name('parameters.json').read_text()))
 def test_plan_undeclared_change_rejected(self):
  p=json.loads(Path(__file__).with_name('parameters.json').read_text());p['algorithm']['gamma']=.99
  with self.assertRaises(AssertionError):validate_plan(p)
 def test_counts_prefill24_and48(self):
  t=Fake()
  for i in range(64):
   run_macro(t)
   if i==15:self.assertEqual(t.counters.critic_updates,0)
   if i in (39,63):self.assertEqual(t.counters.critic_updates,(i+1-16)*128)
  self.assertEqual(t.counters.transitions,65536);self.assertEqual(t.counters.actor_updates,3072)
 def test_parent_learning_config_strict(self):
  config=dict(seed=1201,task={},network={},collection=dict(num_envs=128,steps_per_iteration=8,warmup_iterations=16,capacity_per_env=512),algorithm=dict(actor_lr=5e-5,auxiliary_updates_per_iteration=128,gamma=.97),device='cuda:0')
  state=dict(format='tita-sac-checkpoint',version=1,replay=None,config=config,modules={n:{} for n in MODULES},optimizers={n:{} for n in OPTIMIZERS},action_bounds={},contract_joint_names=[],r6=dict(identity=dict(plan_id='SAC-R6-STABILITY-20261109',direction='R1',fork_seed=1201),entry=dict(parent_sha256='52868414e1d097f011a4da94b45cd8b0267b705708dec1bd45ad3b58470319f8')),counters=dict(iterations=64,env_steps=512,transitions=65536,critic_updates=6144,actor_updates=3072,alpha_updates=6144,target_updates=6144,auxiliary_updates=6144,auxiliary_skipped=0,learning_skipped_iterations=0))
  current=copy.deepcopy(config);current['seed']=1401
  validate_parent(state,current,{},[],{},'P0',1401)
  current['algorithm']['actor_lr']=.0002
  with self.assertRaises(AssertionError):validate_parent(state,current,{},[],{},'P0',1401)
 def test_selection_checks_actual_parent_settings(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);settings=dict(parking_body_velocity_weight=-8,command_resampling_time_range=[2,4],track_ang_vel_z_exp_weight=.5,actor_lr=5e-5,freeze_representation=False)
   write(root/'settings.json',settings)
   s=dict(plan_id=PLAN,selected=dict(model_path='/absolute/model.pt',model_sha256=PARENT_SHA,bytes=PARENT_BYTES,direction='R1',fork_seed=1201,learning_macros=48,source_config={k:settings[k] for k in ('parking_body_velocity_weight','command_resampling_time_range','track_ang_vel_z_exp_weight')}),parent_effective_settings_path='settings.json',parent_effective_settings_sha256=hashlib.sha256((root/'settings.json').read_bytes()).hexdigest())
   write(root/'selection.json',s);selection_record(root/'selection.json',root)
   settings['parking_body_velocity_weight']=0;write(root/'settings.json',settings)
   with self.assertRaises(AssertionError):selection_record(root/'selection.json',root)
 def test_four_distinct_lane_barrier(self):
  with tempfile.TemporaryDirectory() as d:
   identity=dict(plan_id=PLAN,direction='P0',fork_seed=1401,parameters_sha256='same');req=dict(execution_mode='four_lane',barrier_id='test',barrier_timeout_s=2)
   def controller():
    folder=Path(d)/'barriers/test'
    while not (folder/'P0-1401.ready.json').exists():time.sleep(.01)
    for lane in LANES[1:]:
     arm,seed=lane.split('-');write(folder/(lane+'.ready.json'),dict(identity=dict(identity,direction=arm,fork_seed=int(seed)),counters={'iterations':16},barrier_id='test'))
    write(folder/'release.json',dict(barrier_id='test',ready_sha256={lane:hashlib.sha256((folder/(lane+'.ready.json')).read_bytes()).hexdigest() for lane in LANES}))
   thread=threading.Thread(target=controller);thread.start();barrier_wait(d,req,identity,{'iterations':16},write,lambda:False);thread.join()
 def test_runtime_manager_compute_once_and_reward_add_not_duplicated(self):
  text=Path(__file__).with_name('r7p_train.py').read_text();self.assertEqual(text.count('result=TitaSacHooks.compute_rewards(self)'),1)
  self.assertIn('weight=-direction["new_parking_precision_weight"]',text)
  self.assertIn('torch.testing.assert_close(result,original+extra',text)
  self.assertNotIn('return result+extra',text)
 def test_no_old_preflight_gate_and_nodes(self):
  text=Path(__file__).with_name('r7p_train.py').read_text();self.assertNotIn('cfg["preflight_passed"]',text)
  self.assertIn('c["iterations"]-16 in (24,48)',text);self.assertIn('>=300:',text)
  tree=ast.parse(text);app=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='AppLauncher')
  regs=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='signal.signal'];self.assertEqual(len(regs),4);self.assertTrue(all(x>app for x in regs))
if __name__=='__main__':unittest.main()
