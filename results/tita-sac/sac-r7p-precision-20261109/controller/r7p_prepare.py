"""最小配置生成；最新用户已覆盖旧预检/预算阻塞，不启动额外预检或归档。"""
import argparse,json,shlex,time,os
from pathlib import Path
from r7p_ops import *

def prepare(root,trainer_code,r6_root,worker_template,child_env_config,backup_command=None,python_executable=None):
 root=Path(root).resolve();code=Path(trainer_code).resolve();r6=Path(r6_root).resolve();root.mkdir(parents=True,exist_ok=True)
 if (root/'runtime.json').exists() or (root/'controller-state.json').exists():raise ValueError('Existing R7P run; do not overwrite')
 parameters=code/'parameters.json';p=read(parameters)
 if p['plan_id']!=PLAN or p['parent']['sha256']!=PARENT:raise ValueError('Wrong R7P plan/parent')
 parent=r6/'arms/R1-1201/model-learning-48.pt';effective=r6/'arms/R1-1201/effective-settings.json'
 if sha(parent)!=PARENT or parent.stat().st_size!=9167667:raise ValueError('Parent identity mismatch')
 linked=root/'inputs/parent.pt';linked.parent.mkdir(parents=True,exist_ok=True)
 if linked.exists():
  if not os.path.samefile(linked,parent):raise ValueError('Existing parent link differs')
 else:os.link(parent,linked)
 if sha(linked)!=PARENT:raise ValueError('Linked parent SHA mismatch')
 # Consumers read only; no chmod/write to shared inode or original checkpoint.
 parent=linked
 selection=dict(plan_id=PLAN,selected=dict(model_path=str(parent),model_sha256=PARENT,bytes=9167667,direction='R1',fork_seed=1201,learning_macros=48,
  source_config=dict(parking_body_velocity_weight=-8.,command_resampling_time_range=[2.,4.],track_ang_vel_z_exp_weight=.5)),
  parent_effective_settings_path=str(effective),parent_effective_settings_sha256=sha(effective))
 atomic(root/'parameters.json',p);atomic(root/'parent-selection.json',selection)
 override=dict(plan_id=PLAN,recorded_epoch=time.time(),source='latest_explicit_user_instruction',user_messages=['？预检和归档了，直接四路开始R7P，这是命令','别预检和归档了','做好节点训练信息保存','每30分钟保存一次训练信息'],
  reviewed_plan_sha256=sha(parameters),superseded_launch_restrictions=['preflight/archive-only status','old budget/preflight launch gate','future-window-only restriction','prediction-only early stopping'],
  scientific_parameters_unchanged=True,gpu_preflight_status='not_run_by_latest_user_instruction',direct_formal_authorized=True,
  checkpoint_nodes_learning_macros=[24,48],fullresume_interval_seconds=300,progress_interval_seconds=1800,lease_expires_at_utc='2026-11-09T14:15:00Z',soft_target_utc='2026-11-09T12:45:00Z',auto_renew=False)
 atomic(root/'user-execution-override.json',override)
 cfg=read(worker_template)
 for key in ('preflight_passed','preflight_reports','preflight_report'):cfg.pop(key,None)
 cfg.update(lock_file=str(root/'training-init.lock'),use_cache=False)
 atomic(root/'worker-config.json',cfg)
 envcfg=read(child_env_config);env=dict(envcfg['expected_env']);env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 python=python_executable or cfg.get('python') or '/root/rivermind-data/tita-check/.venv/bin/python'
 command=[python,str(code/'r7p_train.py'),'--worker-config',str(root/'worker-config.json'),'--headless','--device','cuda:0']
 sources=[code/name for name in ('r7p_train.py','r7p_training.py','r7p_counts.py')]
 sources.extend([root/'parameters.json',root/'parent-selection.json',root/'worker-config.json',root/'user-execution-override.json',effective,parent])
 runtime=dict(plan_id=PLAN,workroot=str(root),python=python,worker_command=command,direct_formal_authorized=True,
  barrier_id='r7p-four-'+str(time.time_ns()),launch_stagger_seconds=2,lease_expires_at_utc=override['lease_expires_at_utc'],soft_target_utc=override['soft_target_utc'],
  lease_source='latest_user_report_no_renewal',child_cwd=cfg['root'],child_env=env,child_env_remove=['LD_LIBRARY_PATH','LD_PRELOAD'],
  frozen_inputs=[file_record(path) for path in sources],backup_command=backup_command,
  preflight='not_run_per_user_override',old_budget_gate='superseded_by_latest_direct_command',evaluation_cases_expected=30)
 atomic(root/'runtime.json',runtime)
 launch=[python,str(Path(__file__).with_name('r7p_controller.py')),'--runtime',str(root/'runtime.json')]
 atomic(root/'launch-command.json',dict(argv=launch,shell_command=shlex.join(launch),runtime_sha256=sha(root/'runtime.json')))
 return launch
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--workroot',required=True);a.add_argument('--trainer-code',required=True);a.add_argument('--r6-root',required=True);a.add_argument('--worker-template',required=True);a.add_argument('--child-env-config',required=True);a.add_argument('--backup-command-json');a.add_argument('--python');x=a.parse_args()
 print(json.dumps(prepare(x.workroot,x.trainer_code,x.r6_root,x.worker_template,x.child_env_config,read(x.backup_command_json) if x.backup_command_json else None,x.python),ensure_ascii=False))
