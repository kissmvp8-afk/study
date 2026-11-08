"""CPU协议测试；不声称验证Torch、CUDA、Isaac或实际机器人。"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from budget import allocate
from selection import analyze_rows, expected_counters, rank_direction, select, assess_manifest
from training_helpers import frozen_replay_sampling, run_macro


def rows(moving=True):
    result=[]
    for i in range(600):
        cmd=[0.,0.,0.] if i<100 or 300<=i<400 else ([.3,0.,0.] if i<300 else [-.3,0.,0.])
        result.append(dict(step=i,t_start_s=.02*i,t_end_s=.02*(i+1),effective_command=cmd,
                           measured_body_vx_vy_wz=list(cmd) if moving else [0.,0.,0.],root_pos_w=[0.,0.,.30],
                           wheel_target_rad_s=[0.,0.],contact_named_max_N={'base':0.},
                           terminated=False,time_out=False,termination_terms={'base_contact':False,'time_out':False},
                           external_force_b_command=[[0.,0.,0.]],external_torque_b_command=[[0.,0.,0.]]))
    return result


def cases(good1=True,good2=True):
    result=[]
    for seed,good in ((1001,good1),(1002,good2)):
        for _ in range(6):
            result.append(dict(fork_seed=seed,survived=good,task_pass=good,motion_competent=good,loss=0. if good else 5.))
    return result


class Vector:
    def __init__(self,value=0): self.value=value
    def clone(self): return Vector(self.value)
    def equal(self,other): return self.value==other.value


class Buffer:
    def __init__(self): self._writes=Vector(); self.continuity_calls=0; self.valid_calls=0
    def _continuity(self): self.continuity_calls+=1; return [self._writes.value]
    def valid_starts(self,n): self.valid_calls+=1; return (self._continuity(),n)
    def append(self,batch): self._writes.value+=1
    def load_state_dict(self,state): self._writes.value=state


class Trainer:
    def __init__(self,learning=False):
        self.config=NS(collection=NS(num_envs=128,steps_per_iteration=8,warmup_iterations=16,capacity_per_env=512),
                       algorithm=NS(critic_updates_per_iteration=128,auxiliary_updates_per_iteration=128,batch_size=256,n_steps=5))
        self.counters=NS(**expected_counters(0),auxiliary_updates=0,auxiliary_skipped=0)
        if not learning: self.counters.iterations=0; self.counters.env_steps=0; self.counters.transitions=0
        self.buffer=Buffer(); self.calls=[]
    def collect_steps(self,n):
        self.calls.append('collect'); self.counters.env_steps+=n; self.counters.transitions+=128*n; self.buffer.append(None)
        return {'reward_mean':0.}
    def learn_from_replay(self,n):
        self.calls.append('critic'+str(n))
        for key in ('critic_updates','alpha_updates','target_updates'): setattr(self.counters,key,getattr(self.counters,key)+n)
        self.counters.actor_updates+=n//2
        return {'skipped':False,'critic_loss':1.}
    def auxiliary_update(self): self.calls.append('aux'); self.counters.auxiliary_updates+=1; return {'skipped':False}


def make_arm(root,d,seed,learning):
    folder=root/f'{d}-{seed}-{learning}';folder.mkdir(exist_ok=True)
    model=folder/'model.pt';model.write_bytes(f'model-{d}-{seed}-{learning}'.encode());sha=hashlib.sha256(model.read_bytes()).hexdigest()
    spec=json.loads((Path(__file__).parent/'parameters.json').read_text())
    if not (root/'parameters.json').exists():(root/'parameters.json').write_text(json.dumps(spec))
    effective=spec['common']|next(r for r in spec['directions'] if r['id']==d)
    (folder/'effective.json').write_text(json.dumps(effective))
    cases=[]
    for execution in ('D','S'):
        for eval_seed in (101,102,103):
            sub=folder/f'{execution}-{eval_seed}';sub.mkdir(exist_ok=True)
            cfg=dict(sha256=sha,execution=execution,evaluation_seed=eval_seed,condition='standard',policy_noise='on',
                     zero_reset_wrench=True,interval_push_disabled=True,tf32_matmul=False,tf32_cudnn=False,
                     termination_event_source='current_compute_return',physical_initial_state_sha256='0'*64)
            (sub/'config.json').write_text(json.dumps(cfg));(sub/'frozen.json').write_text('{}')
            (sub/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows()))
            rel=lambda name:str((sub/name).relative_to(root))
            cases.append(dict(execution=execution,eval_seed=eval_seed,configuration_path=rel('config.json'),
                              steps_path=rel('steps.jsonl'),frozen_before_path=rel('frozen.json'),frozen_after_path=rel('frozen.json')))
    return dict(direction=d,fork_seed=seed,learning_macros=learning,counters=expected_counters(learning)|
                dict(auxiliary_updates=128*learning,auxiliary_skipped=0),model_path=str(model.relative_to(root)),
                model_sha256=sha,settings_verified=True,effective_settings_path=str((folder/'effective.json').relative_to(root)),cases=cases)


class ProtocolTests(unittest.TestCase):
    def test_stationary_is_not_motion_success(self):
        still=analyze_rows(rows(False));good=analyze_rows(rows())
        self.assertFalse(still['motion_competent']);self.assertFalse(still['task_pass'])
        self.assertTrue(good['task_pass']);self.assertLess(good['loss'],still['loss'])

    def test_fall_is_failed_with_null_unobserved_stages(self):
        r=rows()[:15];r[-1]['terminated']=True;r[-1]['termination_terms']['base_contact']=True
        result=analyze_rows(r)
        self.assertFalse(result['survived']);self.assertEqual(result['loss'],5.)
        self.assertIsNone(result['metrics']['forward']);self.assertTrue(result['loss_is_failure_penalty'])

    def test_truncated_file_is_not_a_valid_fall(self):
        with self.assertRaisesRegex(ValueError,'Truncated'): analyze_rows(rows()[:15])

    def test_retained_terminal_flags_rejected(self):
        r=rows();r[2]['termination_terms']['base_contact']=True
        with self.assertRaisesRegex(ValueError,'current'): analyze_rows(r)

    def test_reward_does_not_change_score(self):
        r=rows();a=analyze_rows(r)
        for row in r: row['reward']=1e12
        self.assertEqual(a,analyze_rows(r))

    def test_one_lucky_fork_cannot_win_over_two_survivors(self):
        lucky=rank_direction('D1',cases(True,False));steady=rank_direction('D2',cases())
        self.assertLess(steady['sort_key'],lucky['sort_key'])

    def test_control_reserved_only_in_round1(self):
        rankings=[rank_direction(d,cases()) for d in ('D0','D1','D2','D3')]
        for r in rankings: r['sort_key'][0]=0 if r['direction']=='D0' else -1
        self.assertIn('D0',select(rankings,1)[1]);self.assertNotIn('D0',select(rankings,2)[1])

    def test_budget_counter_identity_and_hard_stop(self):
        b=allocate(6.24,25,2400,37200)
        self.assertEqual(b['status'],'ready');q=b['q']
        self.assertEqual(b['total_scheduled_learning_macros'],31*q)
        self.assertEqual(b['new_transitions_per_surviving_arm'][-1],1024*(16+6*q))
        self.assertLessEqual(b['predicted_wall_s'],b['hard_cap_s'])
        self.assertEqual(allocate(6.24,25,2400,1800)['status'],'blocked_insufficient_time_for_fair_minimum')

    def test_frozen_cache_is_exact_and_refreshes_after_collection(self):
        b=Buffer();expected=b.valid_starts(5)
        with frozen_replay_sampling(b):
            self.assertEqual(b.valid_starts(5),expected)
            self.assertEqual(b.valid_starts(5),expected)
            with self.assertRaises(RuntimeError):b.append(None)
            with self.assertRaises(RuntimeError):b.load_state_dict(2)
        b.append(None)
        with frozen_replay_sampling(b):self.assertEqual(b.valid_starts(5),([1],5))
        self.assertNotIn('valid_starts',b.__dict__)

    def test_cache_restores_on_exception(self):
        b=Buffer()
        with self.assertRaisesRegex(RuntimeError,'test'):
            with frozen_replay_sampling(b):raise RuntimeError('test')
        b.append(None);self.assertFalse(hasattr(b,'_night_sampling_frozen'))

    def test_macro_preserves_eight_aux_refresh_boundaries(self):
        t=Trainer(learning=True);run_macro(t)
        self.assertEqual(t.calls,['collect']+(['critic16']+['aux']*16)*8)
        self.assertEqual(t.counters.critic_updates,128);self.assertEqual(t.counters.actor_updates,64)
        w=Trainer();run_macro(w);self.assertEqual(w.calls,['collect'])

    def test_manifest_rejects_unequal_budget_and_wrong_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm1=make_arm(root,'D0',1001,64);arm2=make_arm(root,'D0',1002,64)
            data=dict(plan_id='SAC-R4-128-SCREEN-20261108',round=1,q=64,expected_directions=['D0'],arms=[arm1,arm2])
            p=root/'manifest.json';p.write_text(json.dumps(data));self.assertEqual(assess_manifest(p)['promoted'],['D0'])
            data['arms'][1]['counters']['critic_updates']-=1;p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Counter mismatch'):assess_manifest(p)
            data['arms'][1]['counters']['critic_updates']+=1;data['arms'][1]['model_sha256']='0'*64;p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):assess_manifest(p)

    def test_controller_runs_all_rounds_and_does_not_repeat_terminal_task(self):
        import datetime as dt
        import night_controller as controller
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);now=dt.datetime.now(dt.timezone.utc)
            (root/'budget.json').write_text(json.dumps(allocate(6.24,25,0,39600)))
            (root/'parent.json').write_text(json.dumps({'sort_key':[-1.,0.,0.,0.,0.,1.,1.,0]}))
            runtime=dict(workroot=str(root),budget_path='budget.json',worker_command=['fake-worker'],
                         batch_started_at_utc=now.isoformat(),lease_expires_at_utc=(now+dt.timedelta(hours=11)).isoformat(),
                         eval_case_seconds=25,gpu_preflight_passed=True,parameters_path='parameters.json',parent_reference_path='parent.json')
            rp=root/'runtime.json';rp.write_text(json.dumps(runtime));calls=[]
            class FakeProcess:
                def __init__(self,argv,**kwargs):
                    request=json.loads(Path(argv[-3]).read_text());response=Path(argv[-1]);calls.append(request['action'])
                    if request['action']=='train_evaluate':
                        result={'arm':make_arm(root,request['direction'],request['fork_seed'],request['target_learning_macros'])}
                    else:result={'simulated_test_only':True}
                    response.write_text(json.dumps(result));self.returncode=0
                def poll(self):return 0
            with patch.object(controller.subprocess,'Popen',FakeProcess), patch.object(controller.shutil,'disk_usage',return_value=NS(free=60*2**30)):
                controller.main(rp)
                state=json.loads((root/'controller-state.json').read_text())
                self.assertEqual(state['status'],'completed');self.assertEqual(state['completed_round'],4)
                self.assertEqual(state['extra_chunks_completed'],2)
                self.assertEqual(calls.count('train_evaluate'),28)
                previous=len(calls);controller.main(rp);self.assertEqual(len(calls),previous)


if __name__=='__main__':unittest.main()
