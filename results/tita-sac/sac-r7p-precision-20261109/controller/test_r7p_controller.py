import tempfile,unittest
from pathlib import Path
from r7p_ops import *
from r7p_controller import release,progress,check_exited_workers
class Tests(unittest.TestCase):
 def test_four_lane_prefill_barrier(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);token='fixture';self.assertFalse(release(root,token,{}))
   for direction,fork in ARMS:
    atomic(root/'barriers'/token/(direction+'-'+str(fork)+'.ready.json'),dict(barrier_id=token,identity=dict(plan_id=PLAN,direction=direction,fork_seed=fork,parameters_sha256='same',parent_selection_sha256='same',execution_mode='four_lane'),counters=dict(iterations=16,critic_updates=0,actor_updates=0,alpha_updates=0,target_updates=0,auxiliary_updates=0,auxiliary_skipped=0)))
   self.assertTrue(release(root,token,{}));self.assertEqual(len(read(root/'barriers'/token/'release.json')['ready_sha256']),4)
   p=root/'barriers'/token/'P1-1402.ready.json';x=read(p);x['counters']['critic_updates']=1;atomic(p,x)
   with self.assertRaises(ValueError):release(root,token,{})
 def test_progress_has_all_lanes_and_nodes(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);runtime=dict(soft_target_utc='2026-11-09T12:45:00Z',lease_expires_at_utc='2026-11-09T14:15:00Z')
   atomic(root/'runtime.json',runtime);atomic(root/'parameters.json',{})
   for direction,fork in ARMS:
    a=root/'arms'/(direction+'-'+str(fork));a.mkdir(parents=True);(a/'model-learning-24.pt').write_bytes(b'x');atomic(a/'latest-fullresume.json',dict(counters=dict(iterations=40)))
   value=progress(root,runtime,0,'running');self.assertEqual(len(value['arms']),4);self.assertTrue(all(x['model_nodes_present']==[24] for x in value['arms'].values()))
 def test_exit0_without_trained_response_stops(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as d:
   response=Path(d)/'response.json';process=SimpleNamespace(poll=lambda:0)
   with self.assertRaisesRegex(RuntimeError,'returncode.*0'):check_exited_workers([(process,'P0-1401',response)])
   atomic(response,{'status':'failed'})
   with self.assertRaises(RuntimeError):check_exited_workers([(process,'P0-1401',response)])
   atomic(response,{'status':'trained'});check_exited_workers([(process,'P0-1401',response)])
 def test_source_mutation_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'code.py';p.write_text('a');rows=[file_record(p)];verify(rows);p.write_text('b')
   with self.assertRaises(ValueError):verify(rows)
if __name__=='__main__':unittest.main()
