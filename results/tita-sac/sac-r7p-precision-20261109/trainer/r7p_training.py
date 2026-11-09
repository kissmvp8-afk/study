"""R6受限暖启动、冻结束和计数；不修改共享SAC公式及严格restore。"""
import copy, hashlib, json, time
from pathlib import Path
from dataclasses import asdict, replace
from r7p_counts import counts
PLAN='SAC-R7P-PRECISION-PILOT-20261109'
PARENT_SHA='cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3'
PARENT_BYTES=9167667
ARMS={n:dict(actor_lr=5e-5,freeze_representation=False,auxiliary_attempts_per_learning_macro=128,auxiliary_attempts_per_microcycle=16,new_parking_precision_weight=w) for n,w in [('P0',0.),('P1',.5)]}
LANES=('P0-1401','P0-1402','P1-1401','P1-1402')
MODULES=('encoder','projector','actor','critics','targets','temperature')
OPTIMIZERS=('critic','actor','alpha','auxiliary')
def canonical(x):return json.loads(json.dumps(x))
def selection_record(path,workroot=None):
    s=json.loads(Path(path).read_text());p=s['selected']
    assert s['plan_id']==PLAN
    assert (p['model_sha256'],p['bytes'],p['direction'],p['fork_seed'],p['learning_macros'])==(PARENT_SHA,PARENT_BYTES,'R1',1201,48)
    model=Path(p['model_path']);root=Path(workroot).resolve()
    if not model.is_absolute():model=(root/model).resolve();assert model.is_relative_to(root)
    s['resolved_model_path']=str(model.resolve())
    assert p['source_config']==dict(parking_body_velocity_weight=-8,command_resampling_time_range=[2,4],track_ang_vel_z_exp_weight=.5)
    evidence=Path(s['parent_effective_settings_path'])
    if not evidence.is_absolute():evidence=(root/evidence).resolve();assert evidence.is_relative_to(root)
    assert hashlib.sha256(evidence.read_bytes()).hexdigest()==s['parent_effective_settings_sha256']
    settings=json.loads(evidence.read_text())
    for k,v in p['source_config'].items():assert settings[k]==v,('parent environment mismatch',k)
    assert settings['actor_lr']==5e-5 and settings['freeze_representation'] is False
    s['resolved_parent_effective_settings_path']=str(evidence)
    return s

def inherited_direction(selection,arm):
    return dict(id=arm,**ARMS[arm],alpha_entry_multiplier=1,track_ang_vel_z_exp_weight=.5,
      parking_body_velocity_weight=-8,command_resampling_time_range=[2,4])

def configure_arm(trainer,arm):
    a=ARMS[arm]
    trainer.config=replace(trainer.config,algorithm=replace(trainer.config.algorithm,
      actor_lr=a['actor_lr'],auxiliary_updates_per_iteration=a['auxiliary_attempts_per_learning_macro']))

def validate_parent(state,current,bounds,joints,selection,arm,seed):
    old=canonical(state['config']);current=canonical(current)
    assert state['format']=='tita-sac-checkpoint' and state['version']==1 and state['replay'] is None
    assert set(state['modules'])==set(MODULES) and set(state['optimizers'])==set(OPTIMIZERS)
    assert seed in (1401,1402) and current['seed']==seed and old['seed']==1201
    assert state['action_bounds']==bounds and state['contract_joint_names']==list(joints)
    identity=state['r6']['identity']
    assert identity['plan_id']=='SAC-R6-STABILITY-20261109' and identity['direction']=='R1' and identity['fork_seed']==1201
    assert state['r6']['entry']['parent_sha256']=='52868414e1d097f011a4da94b45cd8b0267b705708dec1bd45ad3b58470319f8'
    for key in set(old)|set(current):
        if key not in ('name','sources','environment','seed'):assert current[key]==old[key],('undeclared_config_change',key)
    assert current['collection']==dict(num_envs=128,steps_per_iteration=8,warmup_iterations=16,capacity_per_env=512)
    assert current['algorithm']['actor_lr']==5e-5 and current['algorithm']['auxiliary_updates_per_iteration']==128
    c=state['counters']
    for key,value in dict(iterations=64,env_steps=512,transitions=65536,critic_updates=6144,actor_updates=3072,alpha_updates=6144,target_updates=6144,auxiliary_updates=6144,auxiliary_skipped=0,learning_skipped_iterations=0).items():assert c[key]==value,key

def warm_start_r7p(trainer,parent_path,seed,selection,arm,fingerprint):
    import torch
    from checkpoint import _bounds_record
    p=selection['selected'];path=Path(parent_path)
    assert p['model_sha256']==PARENT_SHA and p['bytes']==PARENT_BYTES
    assert path.resolve()==Path(selection.get('resolved_model_path',p['model_path'])).resolve()
    assert path.stat().st_size==p['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==p['model_sha256']
    state=torch.load(path,map_location='cpu',weights_only=True)
    validate_parent(state,asdict(trainer.config),_bounds_record(trainer),trainer.contract.joint_names,selection,arm,seed)
    assert trainer.buffer.size==0 and not any(counts(trainer).values())
    assert not bool(trainer.observation.episode_id.any()) and not bool(trainer.observation.step_index.any())
    for n in MODULES:
        getattr(trainer,n).load_state_dict(state['modules'][n],strict=True)
        assert fingerprint(getattr(trainer,n).state_dict())==fingerprint(state['modules'][n])
    for n in OPTIMIZERS:
        opt=trainer.optimizers[n];opt.load_state_dict(state['optimizers'][n])
        assert fingerprint(opt.state_dict())==fingerprint(state['optimizers'][n])
        assert all(g['lr']==getattr(trainer.config.algorithm,n+'_lr') for g in opt.param_groups)
    trainer.normalizers.load_statistics(state['normalizers']);trainer.encoder.eval()
    assert fingerprint(trainer.normalizers.export_statistics())==fingerprint(state['normalizers'])
    for offset,n in enumerate(('collect','replay','learn'),1):trainer.generators[n].manual_seed(seed+offset)
    alpha=float(trainer.temperature.log_alpha.detach().exp().item())
    assert abs(alpha/7.544342952314764e-5-1)<1e-6
    return dict(mode='R7P_fixed_R6_parent',parent_sha256=p['model_sha256'],parent_counters=state['counters'],
      parent_alpha=alpha,effective_entry_alpha=alpha,alpha_multiplier=1.,native_learned_std_restored=True,
      inherited_optimizer_steps=True,task_counters_start_at_zero=True,parent_replay_imported=False,
      parent_episode_ids_imported=False,fork_seed=seed,exact_resume=False)

def assert_r7p_counts(trainer):
    c=counts(trainer);k=c['iterations'];n=max(0,k-16)
    for key,v in dict(env_steps=8*k,transitions=1024*k,critic_updates=128*n,alpha_updates=128*n,target_updates=128*n,actor_updates=64*n,learning_skipped_iterations=0).items():assert c[key]==v,key
    assert c['auxiliary_updates']+c['auxiliary_skipped']==trainer.config.algorithm.auxiliary_updates_per_iteration*n
    return c

def run_macro(trainer,use_cache=False):
    import math
    assert not use_cache
    c=trainer.config.collection;a=trainer.config.algorithm
    assert (c.num_envs,c.steps_per_iteration,c.warmup_iterations,c.capacity_per_env)==(128,8,16,512)
    assert (a.critic_updates_per_iteration,a.batch_size,a.n_steps,a.actor_update_every,a.alpha_update_every,a.target_update_every)==(128,256,5,2,1,1)
    assert a.auxiliary_updates_per_iteration in (0,128)
    before=assert_r7p_counts(trainer);learning=before['iterations']>=16
    log=dict(macro_iteration=before['iterations']+1,collect=trainer.collect_steps(8),microcycles=[])
    if learning:
        for _ in range(8):
            learned=trainer.learn_from_replay(16);assert not learned.get('skipped',False)
            auxiliary=[trainer.auxiliary_update() for _ in range(a.auxiliary_updates_per_iteration//8)]
            log['microcycles'].append(dict(learn=learned,auxiliary=auxiliary))
    trainer.counters.iterations+=1;after=assert_r7p_counts(trainer)
    json.dumps(log,allow_nan=False)
    log.update(counters=after,auxiliary_attempts_actual=after['auxiliary_updates']+after['auxiliary_skipped']-before['auxiliary_updates']-before['auxiliary_skipped'],
      auxiliary_success_actual=after['auxiliary_updates']-before['auxiliary_updates'],auxiliary_skipped_actual=after['auxiliary_skipped']-before['auxiliary_skipped'],
      auxiliary_disabled_reason='frozen_representation_and_statistics' if a.auxiliary_updates_per_iteration==0 else None)
    return log

def barrier_wait(root,request,identity,counters,write,should_stop):
    assert request['execution_mode']=='four_lane'
    token=request['barrier_id'];assert token and '/' not in token and '..' not in token
    directory=Path(root)/'barriers'/token;directory.mkdir(parents=True,exist_ok=True)
    lane=identity['direction']+'-'+str(identity['fork_seed']);assert lane in LANES
    path=directory/(lane+'.ready.json');assert not path.exists(),'stale ready file'
    write(path,dict(identity=identity,counters=counters,barrier_id=token))
    deadline=time.monotonic()+request.get('barrier_timeout_s',600)
    while True:
        if should_stop() or time.monotonic()>deadline:raise TimeoutError('R7P learning barrier stopped')
        release=directory/'release.json'
        if release.exists():
            data=json.loads(release.read_text());assert data['barrier_id']==token and set(data['ready_sha256'])==set(LANES)
            common={k:v for k,v in identity.items() if k not in ('direction','fork_seed')}
            for other,digest in data['ready_sha256'].items():
                rp=directory/(other+'.ready.json');assert hashlib.sha256(rp.read_bytes()).hexdigest()==digest
                item=json.loads(rp.read_text());i=item['identity']
                assert i['direction']+'-'+str(i['fork_seed'])==other
                assert {k:v for k,v in i.items() if k not in ('direction','fork_seed')}==common
                assert item['counters']['iterations']==counters['iterations']
            return
        time.sleep(.2)

def precision_values(command,linear,angular):
    import torch
    mask=(torch.linalg.vector_norm(command[:,:2],dim=1)<.001)&(command[:,2].abs()<.001)
    cost=-torch.expm1(-linear[:,:2].square().sum(1)/.1**2)+.25*(-torch.expm1(-angular[:,2].square()/.2**2))
    return mask,cost

def parking_precision_cost(env):
    d=env.scene['robot'].data
    mask,cost=precision_values(env.command_manager.get_command('base_velocity'),d.root_lin_vel_b,d.root_ang_vel_b)
    return mask.to(cost.dtype)*cost

EFFECTIVE_COMMON={'num_envs': 128, 'steps_per_macro': 8, 'warmup_macros': 16, 'capacity_per_env': 512, 'replay_capacity_total': 65536, 'critic_updates_per_learning_macro': 128, 'actor_updates_per_learning_macro': 64, 'alpha_updates_per_learning_macro': 128, 'target_updates_per_learning_macro': 128, 'microcycles': 8, 'critic_updates_per_microcycle': 16, 'batch_size': 256, 'auxiliary_batch_size': 256, 'gamma': 0.97, 'n_steps': 5, 'tau': 0.003, 'critic_lr': 0.0002, 'alpha_lr': 2e-05, 'auxiliary_lr': 0.001, 'standing_command_fraction': 0.2, 'policy_observation_noise': 'on', 'reset_force_N': [0, 0], 'reset_torque_Nm': [0, 0], 'interval_push': 'off', 'physics_dt_s': 0.005, 'decimation': 4, 'control_dt_s': 0.02, 'tf32_matmul': False, 'tf32_cudnn': False, 'alpha_compute': 'GPU_tensor_common_to_all', 'network_reward_action_bounds_timeout_semantics': 'R4_except_declared_parking_weight_command_duration_and_logging', 'initial_std': 'restore_learned_value_do_not_reset_or_scale_for_evaluation'}

def validate_plan(p):
    assert p['plan_id']==PLAN
    assert (p['parent']['sha256'],p['parent']['bytes'])==(PARENT_SHA,PARENT_BYTES)
    assert p['fork_seeds']==[1401,1402]
    assert p['directions']==[dict(id='P0',meaning='actual-parent-configuration control',new_parking_precision_weight=0.),dict(id='P1',meaning='same actual baseline plus precision term',new_parking_precision_weight=.5)]
    env=p['common_training_environment']
    for k,v in dict(num_envs_per_worker=128,concurrent_workers=4,capacity_per_env=512,replay_capacity_total=65536,existing_parking_body_velocity_weight=-8.,command_resampling_time_range_s=[2.,4.],standing_command_fraction=.2,observation_noise='on',reset_wrench='zero',interval_push='off',TF32='off',use_cache=False).items():assert env[k]==v,k
    for k,v in dict(actor_lr=5e-5,critic_lr=2e-4,alpha_lr=2e-5,auxiliary_lr=.001,gamma=.97,n_steps=5,tau=.003,max_grad_norm=1.,batch_size=256,actor_update_every=2,retain_parent_alpha_and_std=True).items():assert p['algorithm'][k]==v,k
    assert p['schedule']['prefill_macros_per_arm']==16 and p['schedule']['learning_macros_per_arm']==48
    # Earlier budget-only status is superseded by the user's latest direct start instruction;
    # no numeric, parent, reward, sampling or learning constraint is relaxed.
