"""最小CPU契约测试，不启动训练/GPU。"""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evaluation_r7p_helpers import validate_seed,schedule
from r4_protocol import command
from r7p_evaluation import *
def case(p=.04,passed=True,motion=True):
 return dict(steps=600,loss_is_failure_penalty=False,survived=True,task_pass=passed,motion_competent=motion,metrics={'stop1':{'speed':p,'abs_wz':.02},'stop2':{'speed':p,'abs_wz':.02},'forward':{'vx_mae':.04},'reverse':{'vx_mae':.04}})
def arms(p0=.1,p1=.04):
 return [dict(direction=d,fork_seed=f,learning_macros=48,model_path=d+str(f)+'.pt',model_sha256='a'*64,results=[case(p0 if d=='P0' else p1,passed=False) for _ in range(6)]) for d in ('P0','P1') for f in (1401,1402)]
class Contracts(unittest.TestCase):
 def test_only_fixed_development(self):
  for seed in (141,142,143):validate_seed('development',seed,'standard')
  for split,seed,protocol in [('primary',231,'standard'),('development',121,'standard'),('development',141,'turn'),('development',141,'pulse')]:
   with self.assertRaises(AssertionError):validate_seed(split,seed,protocol)
 def test_original_commands_unchanged(self):
  for i in range(600):self.assertEqual(list(schedule('standard',i)[1]),command('standard',i))
 def test_exact30_and_parent_first(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);parent=root/'parent.pt'
   with parent.open('wb') as f:f.truncate(9167667)
   calls=[]
   def callback(*args):calls.append(args);return dict(execution=args[4],eval_seed=args[5])
   def validate(root,item,cases,paired):return [dict(case(),execution=c['execution'],eval_seed=c['eval_seed']) for c in cases]
   with patch('r7p_evaluation.validate_cases',side_effect=validate):result=run_endpoint(root,dict(model_path='parent.pt',model_sha256=PARENT_SHA),arms(),callback)
   self.assertEqual(len(calls),30);self.assertTrue(all(c[0]=='parent.pt' for c in calls[:6]));self.assertEqual(result['total_cases'],30)
   self.assertFalse(result['decision']['independent_acceptance']);self.assertFalse(result['decision']['promotion'])
 def test_promising_and_motion_gate(self):
  aa=arms();p=fork_summary([case(.05) for _ in range(6)])
  self.assertTrue(diagnostic_decision(p,aa)['promising_short_budget'])
  aa[-1]['results'][0]['motion_competent']=False
  self.assertFalse(diagnostic_decision(p,aa)['promising_short_budget'])
 def test_zero_control_no_percent(self):
  result=diagnostic_decision(fork_summary([case(0) for _ in range(6)]),arms(0,0))
  self.assertFalse(result['benefit']['parking_percentage_eligible']);self.assertFalse(result['promising_short_budget'])
 def test_incomplete_endpoint_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'parent';p.touch()
   with p.open('wb') as f:f.truncate(9167667)
   with self.assertRaises(ValueError):run_endpoint(tmp,dict(model_path='parent',model_sha256=PARENT_SHA),arms()[:3],None)
 def test_fall_never_zero_filled(self):
  c=[case() for _ in range(6)];c[0]['loss_is_failure_penalty']=True
  self.assertIsNone(fork_summary(c)['parking_P'])
if __name__=='__main__':unittest.main()
