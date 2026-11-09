"""模型备份调度离线测试，假checkpoint元数据，不连接远端、不声称torch验证。"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import r7p_model_backup_loop as m
from r7p_backup_ops import atomic,sha

def state(arm='P0',fork=1401,stage=48,kind='model',parameters='p',parent='s'):
 c=dict(iterations=stage+16,env_steps=8*(stage+16),transitions=1024*(stage+16),critic_updates=128*stage,alpha_updates=128*stage,target_updates=128*stage,actor_updates=64*stage,auxiliary_updates=128*stage if arm in ('P0','P1') else 0,auxiliary_skipped=0)
 return dict(format='tita-sac-checkpoint',version=1,modules={k:{} for k in ('encoder','projector','actor','critics','targets','temperature')},counters=c,replay={} if kind=='resume' else None,r7p=dict(identity=dict(plan_id=m.PLAN,direction=arm,fork_seed=fork,parameters_sha256=parameters,parent_selection_sha256=parent,execution_mode='serial'),actual_update_counts={k:c[v] for k,v in [('critic','critic_updates'),('actor','actor_updates'),('alpha','alpha_updates'),('target','target_updates'),('auxiliary','auxiliary_updates')]},use_cache=False,entry=dict(parent_sha256='parentmodel')))
class Tests(unittest.TestCase):
 def test_deploy_before_formal_models(self):
  with tempfile.TemporaryDirectory() as d:
   result=m.tick(dict(workroot=d,model_backup_root=str(Path(d)/'backups')))
   self.assertEqual(result['status'],'no_formal_models_yet')
 def test_exact_paths(self):
  items=list(m.expected_files('/tmp/run'));self.assertEqual(len(items),20);self.assertEqual(sum(x['kind']=='resume' for x in items),12)
 def test_identity_and_counts(self):
  item=dict(direction='P0',fork_seed=1401,stage=48,kind='model');s=state();self.assertEqual(m.validate_metadata(s,item,'p','s','serial')['learning_macros'],48)
  for mutate in ('fork','count','replay'):
   bad=copy.deepcopy(s)
   if mutate=='fork':bad['r7p']['identity']['fork_seed']=1402
   if mutate=='count':bad['counters']['actor_updates']-=1
   if mutate=='replay':bad['replay']={}
   with self.assertRaises(ValueError):m.validate_metadata(bad,item,'p','s','serial')
 def test_link_cleanup_never_unique(self):
  import os
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'source';b=Path(d)/'staged';a.write_text('a');os.link(a,b)
   self.assertTrue(m.remove_redundant_link(b,a));self.assertTrue(a.exists())
   os.link(a,b);tmp=Path(d)/'new';tmp.write_text('new');os.replace(tmp,a)
   self.assertFalse(m.remove_redundant_link(b,a));self.assertEqual(b.read_text(),'a')
 def test_loop_incremental_and_pending(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);atomic(root/'parameters.json',{'plan_id':m.PLAN});atomic(root/'parent-selection.json',{'selected':{'model_sha256':'parentmodel'}})
   cfg=dict(workroot=str(root),model_backup_root=str(root/'backups'),parameters_path='parameters.json',parameters_sha256=sha(root/'parameters.json'),parent_selection_sha256=sha(root/'parent-selection.json'),execution_mode='serial',base_commit='a'*40,gh='NO_REMOTE')
   source=root/'arms/P0-1401/model-learning-48.pt';source.parent.mkdir(parents=True);source.write_bytes(b'fake-model')
   loader=lambda p:state(parameters=cfg['parameters_sha256'],parent=cfg['parent_selection_sha256'])
   with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)),patch.object(m,'publish',side_effect=RuntimeError('uncertain')):
    with self.assertRaises(RuntimeError):m.tick(cfg,loader)
   self.assertTrue((root/'backups/pending.sidecar.json').exists())
   with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)),patch.object(m,'publish',return_value={'status':'remote_verified'}) as pub:
    result=m.tick(cfg,loader);self.assertEqual(pub.call_count,1);self.assertEqual(len(result['uploaded']),1)
    result=m.tick(cfg,loader);self.assertEqual(pub.call_count,1);self.assertEqual(result['uploaded'],[])
   self.assertTrue(source.exists())
if __name__=='__main__':unittest.main()
