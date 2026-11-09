"""Frozen native SAC inference on TRAIN task, fixed commands, pre-reset signals; optional simulation-time video."""
import sys,json,hashlib,time,traceback,argparse,math,random
from pathlib import Path
from linux_common import write_json,sha256,prepare_args,Progress
from r1_metrics import stage_metrics,d_gate
import tensordict,h5py
from evaluation_r7p_helpers import physical_fingerprint, current_events, validate_seed, schedule as protocol_schedule, diagnostic_result

from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/root/rivermind-data/tita-check'));p.add_argument('--study-root',type=Path);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--eval_seed',type=int,required=True);p.add_argument('--video',action='store_true');p.add_argument('--condition',choices=['standard'],default='standard');p.add_argument('--execution',choices=['D','S'],default='D')
p.add_argument('--factor',choices=['F00','F10','F01','F11'],required=True);p.add_argument('--policy-noise',choices=['on','off'],required=True);p.add_argument('--zero-reset-wrench',action='store_true');p.add_argument('--disable-interval-push',action='store_true')
p.add_argument('--seed-split',choices=['development'],default='development')
p.add_argument('--startup-lock',type=Path)
p.add_argument('--pulse-direction',choices=['x+','x-','y+','y-'])
AppLauncher.add_app_launcher_args(p);args=prepare_args(p.parse_args());out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
REPO=args.study_root.resolve() if args.study_root else args.root/'study'
validate_seed(args.seed_split,args.eval_seed,args.condition)
assert {'F00':('on',False),'F10':('off',False),'F01':('on',True),'F11':('off',True)}[args.factor]==(args.policy_noise,args.zero_reset_wrench)
assert args.policy_noise == 'on'
assert args.factor=='F01' and args.zero_reset_wrench and args.disable_interval_push
assert (args.condition == 'pulse') == (args.pulse_direction is not None)
assert args.disable_interval_push or (args.factor=='F00' and args.condition in ('standard','diagnostic_zero','diagnostic_switch') and args.execution=='D')
assert args.condition!='pulse' or (args.factor=='F01' and args.zero_reset_wrench and args.disable_interval_push and args.execution=='D')
progress=Progress(out)
def write(n,v):write_json(out/n,v)
write('metadata.json',{'plan_id':'SAC-R7P-PRECISION-PILOT-20261109','argv':sys.argv,'script':str(Path(__file__).resolve()),'script_sha256':sha256(Path(__file__)),'study_root':str(REPO),'evaluation_seed':args.eval_seed,'condition':args.condition})
app=None;env=None;video=None
try:
    startup_lock=None
    if args.startup_lock:
        import fcntl
        args.startup_lock.parent.mkdir(parents=True,exist_ok=True)
        startup_lock=args.startup_lock.open('a');fcntl.flock(startup_lock,fcntl.LOCK_EX)
    app=AppLauncher({**vars(args),'width':960,'height':540}).app
    import torch,numpy as np,gymnasium as gym,importlib
    random.seed(args.eval_seed);np.random.seed(args.eval_seed);torch.manual_seed(args.eval_seed);torch.cuda.manual_seed_all(args.eval_seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import ddt_lab.tasks
    from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
    for name in ('environment_control/action_contract','environment_control/robot_environment'):sys.path.insert(0,str(REPO/name))
    from baseline_source import load_baseline_action_contract
    from assets import verify_urdf,parse_urdf_joints
    def fingerprint(model):
        h=hashlib.sha256()
        for key,v in model.state_dict().items():h.update(key.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    def serial(x):
        if isinstance(x,torch.Tensor):
            if x.is_floating_point() and not torch.isfinite(x).all():raise FloatingPointError('nonfinite signal')
            return x.detach().cpu().tolist()
        if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [serial(v) for v in x]
        return x
    def schedule(i):return protocol_schedule(args.condition,i)
    class FixedCommand(UniformVelocityCommand):
        def _assign(self,ids):
            _,cmd=schedule(int(self._env.episode_length_buf[0]));self.vel_command_b[ids]=torch.tensor(cmd,device=self.device)
            self.is_heading_env[ids]=False;self.is_standing_env[ids]=False
        def _resample_command(self,ids):self._assign(ids)
        def _update_command(self):self._assign(slice(None))
    assert not args.condition.startswith('diagnostic_') or args.execution=='D'
    sha=sha256(args.checkpoint);state=torch.load(args.checkpoint,weights_only=True,map_location='cpu')
    assert set(state['modules'])=={'encoder','projector','actor','critics','targets','temperature'}
    for module_name,module_state in state['modules'].items():
        for tensor_name,tensor in module_state.items():
            if isinstance(tensor,torch.Tensor):assert torch.isfinite(tensor).all(), 'Nonfinite checkpoint module '+module_name+'.'+tensor_name
    spec=gym.spec('DDT-Velocity-Flat-Tita-v0');cfg=spec.kwargs['env_cfg_entry_point']();cfg.scene.num_envs=1;cfg.seed=args.eval_seed;cfg.sim.device=args.device or 'cuda:0'
    cfg.observations.policy.history_length=0
    original_noise=cfg.observations.policy.enable_corruption
    original_push=str(cfg.events.push_robot)
    if args.disable_interval_push:cfg.events.push_robot=None
    # 关闭噪声不修改物理随机化；配对由完整初态记录核对。
    cfg.observations.policy.enable_corruption=(args.policy_noise=='on')
    original_command=cfg.commands.base_velocity.to_dict();cfg.commands.base_velocity.class_type=FixedCommand
    cfg.viewer.resolution=(960,540)
    if args.video:cfg.viewer.eye=(1.8,-2.5,1.3);cfg.viewer.lookat=(0.,0.,.23)
    urdf=Path(cfg.scene.robot.spawn.asset_path);asset_status=verify_urdf(urdf);joints=parse_urdf_joints(urdf)
    env=gym.make('DDT-Velocity-Flat-Tita-v0',cfg=cfg,render_mode='rgb_array' if args.video else None).unwrapped
    # Preserve environment RNG after startup around shape discovery and random network construction.
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
    for module in ('training_validation/configuration','environment_control/action_bounds','environment_control/observation_interface','networks_learning/history_encoder','networks_learning/actor_distribution','experience_returns/normalization'):
        sys.path.insert(0,str(REPO/module))
    from config import config_from_dict
    from bounds import derive_bounds,calibrate_initial_center
    from encoder import HistoryEncoder,features_from_history
    from actor import SquashedGaussianActor
    from normalization import ObservationNormalizers
    from observations import PolicyHistory,push_frame,validate_observations
    validate_observations('isaac_flat',env.observation_manager.active_terms,env.observation_manager.group_obs_term_dim,33)
    assert cfg.observations.critic.base_lin_vel.scale==2.
    assert state['format']=='tita-sac-checkpoint' and state['version']==1
    scfg=config_from_dict(state['config']);contract=load_baseline_action_contract(REPO)
    bounds=derive_bounds(contract,scfg.task.joint_pos_limits,scfg.task.wheel_speed_ranges)
    assert list(contract.joint_names)==state['contract_joint_names']
    for key in ('joint_names','lower','upper','center','half_width'):assert list(getattr(bounds,key))==state['action_bounds'][key]
    # 只加载推理依赖，不创建经验池、优化器、不触发 Trainer.reset 或改变训练状态。
    encoder=HistoryEncoder(scfg.network.encoder_backbone_dims,scfg.network.encoder_latent_hidden_dim)
    actor=SquashedGaussianActor(bounds,calibrate_initial_center(contract,bounds,scfg.task.initial_targets),hidden_dims=scfg.network.actor_hidden_dims,initial_std=scfg.algorithm.initial_std)
    encoder.load_state_dict(state['modules']['encoder'],strict=True);actor.load_state_dict(state['modules']['actor'],strict=True)
    net=torch.nn.ModuleDict({'encoder':encoder,'actor':actor}).to(cfg.sim.device).eval();net.requires_grad_(False)
    normalizers=ObservationNormalizers(scfg.task.q_dim,eps=scfg.algorithm.normalizer_eps)
    normalizers.load_statistics(state['normalizers'])
    stats_before=serial(normalizers.export_statistics())
    sample_generator=torch.Generator(device=cfg.sim.device).manual_seed(args.eval_seed+100000)
    dims={'actor_feature_dim':52,'history':[10,33],'q_dim':scfg.task.q_dim}
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
    before=fingerprint(net);counts={n:m.count for n,m in net.named_modules() if hasattr(m,'count')}
    assert set(env.termination_manager.active_terms)=={'base_contact','time_out'}
    assert not env.termination_manager.get_term_cfg('base_contact').time_out
    assert env.termination_manager.get_term_cfg('time_out').time_out
    robot=env.scene['robot'];sensor=env.scene['contact_forces'];contract=load_baseline_action_contract(REPO)
    jointids=[robot.joint_names.index(e.joint_name) for e in contract.entries];wheel=[3,7];leg=[0,1,2,4,5,6]
    bodies=['base_link']+[joints[e.joint_name].child for e in contract.entries];bodyids=[sensor.body_names.index(n) for n in bodies]
    contact_names={'base':'base_link','left_leg_3':joints['joint_left_leg_3'].child,'right_leg_3':joints['joint_right_leg_3'].child,'left_wheel':joints['joint_left_leg_4'].child,'right_wheel':joints['joint_right_leg_4'].child}
    contact_indices={k:sensor.body_names.index(v) for k,v in contact_names.items()}
    # 条件之间显式重置同一环境 RNG；不生成额外观测，不为匹配物理状态复制噪声观测。
    random.seed(args.eval_seed);np.random.seed(args.eval_seed);torch.manual_seed(args.eval_seed);torch.cuda.manual_seed_all(args.eval_seed)
    obs,info=env.reset();assert env.episode_length_buf.item()==0
    history=PolicyHistory(1,device=env.device)
    history.history,history.pushes=push_frame(history.history,history.pushes,obs['policy'])
    assert math.isclose(env.step_dt,.02,abs_tol=1e-10), 'Protocol requires original .02 second control dt'
    reset_wrench=serial({'force_b_command':robot._external_force_b,'torque_b_command':robot._external_torque_b,'submission_enabled':bool(robot.has_external_wrench)})
    if args.zero_reset_wrench:
        robot.set_external_force_and_torque(torch.zeros_like(robot._external_force_b),torch.zeros_like(robot._external_torque_b))
        assert torch.count_nonzero(robot._external_force_b)==0 and torch.count_nonzero(robot._external_torque_b)==0
    physics_start=env._sim_step_counter
    pulse_state={'active':False,'next_substep':0,'substeps':[]}
    if args.condition=='pulse':
        assert cfg.decimation==4 and abs(env.physics_dt-.005)<1e-10
        base_body_id=robot.body_names.index('base_link')
        pulse_vector={'x+':(20.,0.,0.),'x-':(-20.,0.,0.),'y+':(0.,20.,0.),'y-':(0.,-20.,0.)}[args.pulse_direction]
        old_scene_write=env.scene.write_data_to_sim
        def pulse_scene_write():
            forces=torch.zeros_like(robot._external_force_b)
            torques=torch.zeros_like(robot._external_torque_b)
            if pulse_state['active']:
                # Isaac Lab 2.3 increments _sim_step_counter BEFORE this hook.
                index=env._sim_step_counter-physics_start-1
                assert index==pulse_state['next_substep']
                if 800<=index<840:
                    forces[0,base_body_id]=torch.tensor(pulse_vector,device=env.device)
            robot.set_external_force_and_torque(forces,torques,is_global=False)
            assert torch.equal(robot._external_force_b,forces) and torch.equal(robot._external_torque_b,torques)
            if pulse_state['active']:
                other=forces.clone();other[:,base_body_id]=0
                pulse_state['substeps'].append(serial({
                    'substep':index,'t_start_s':index*.005,
                    'base_force_b_N':robot._external_force_b[0,base_body_id],
                    'base_torque_b_Nm':robot._external_torque_b[0,base_body_id],
                    'other_body_force_abs_max':other.abs().max(),
                    'all_torque_abs_max':robot._external_torque_b.abs().max(),
                    'is_global':False,'submission_enabled':bool(robot.has_external_wrench)}))
                pulse_state['next_substep']+=1
            # Reset-time writes clear all force/torque too, but are not physics substeps.
            return old_scene_write()
        env.scene.write_data_to_sim=pulse_scene_write
    progress.update('evaluation')
    initial=serial({'root_pos_w':robot.data.root_pos_w,'root_com_pos_w':robot.data.root_com_pos_w,'root_quat_w':robot.data.root_quat_w,
        'root_lin_vel_w':robot.data.root_lin_vel_w,'root_ang_vel_w':robot.data.root_ang_vel_w,
        'material_properties':robot.root_physx_view.get_material_properties(),'masses':robot.root_physx_view.get_masses(),'inertias':robot.root_physx_view.get_inertias(),'coms':robot.root_physx_view.get_coms(),
        'explicit_actuators':{n:{'joint_names':a.joint_names,'stiffness':a.stiffness,'damping':a.damping} for n,a in robot.actuators.items()},
        'joint_pos':robot.data.joint_pos,'joint_vel':robot.data.joint_vel,'stiffness':robot.data.joint_stiffness,'damping':robot.data.joint_damping,
        'external_force_b':robot._external_force_b,'external_torque_b':robot._external_torque_b})
    initial_sha256=physical_fingerprint(initial)
    rng_identity={'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),'numpy':hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),'torch':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'cuda':[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()],'sample_generator':hashlib.sha256(sample_generator.get_state().cpu().numpy().tobytes()).hexdigest()}
    source_identity={str(Path(m.__file__).resolve()):sha256(Path(m.__file__)) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file() and (Path(m.__file__).resolve().is_relative_to(REPO) or Path(m.__file__).resolve().is_relative_to(Path(__file__).resolve().parent))}
    if args.video:
        import cv2
        warm_state=robot.data.root_state_w.detach().clone()
        pos=robot.data.root_pos_w[0].tolist();env.sim.set_camera_view((pos[0]+1.8,pos[1]-2.5,1.3),(pos[0],pos[1],.23))
        # Warm rendering only; never advance physics or compute more observations.
        env.render()
        for _ in range(6):env.sim.render()
        assert env._sim_step_counter==physics_start and torch.equal(warm_state,robot.data.root_state_w),'render warmup changed physics state'
        video=cv2.VideoWriter(str(out/f'SAC-{args.execution}-seed{args.eval_seed}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(960,540));assert video.isOpened()
    write('configuration.json',{'checkpoint':str(args.checkpoint),'sha256':sha,'checkpoint_counters':state['counters'],'execution':args.execution,'evaluation_noise_seed':args.eval_seed+100000,'checkpoint_learning_device':scfg.device,'training_seed':state['config'].get('seed', None),'evaluation_seed':args.eval_seed,
        'plan_id':'SAC-R7P-PRECISION-PILOT-20261109','condition':args.condition,'argv':sys.argv,'original_policy_noise':original_noise,'original_push':original_push,
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'termination_event_source':'current_compute_return','physical_initial_state_sha256':initial_sha256,'seed_split':args.seed_split,
        'reset_rng_seed':args.eval_seed,'factor':args.factor,'policy_noise':args.policy_noise,'zero_reset_wrench':args.zero_reset_wrench,'interval_push_disabled':args.disable_interval_push,'reset_wrench_before_intervention':reset_wrench,'contact_names':contact_names,
        'pulse_protocol':({'direction':args.pulse_direction,'force_body_N':20,'torque_Nm':0,'start_s':4.,'end_s':4.2,'substep_range':[800,840],'command_impulse_Ns':4.} if args.condition=='pulse' else None),
        'wrench_semantics':'command and submission flag, NOT measured total external force',
        'task':'DDT-Velocity-Flat-Tita-v0','Play_task_used':False,'env':cfg.to_dict(),'original_command':original_command,'dimensions':dims,
        'inference':'SAC native features_from_history + deterministic_action or original std sample_action_and_log_prob; frozen encoder/actor/normalizers; independent sampling generator',
        'rollout_ready_unix':time.time(),'rng_identity':rng_identity,'source_identity':source_identity,'initial_state':initial,'asset':{'path':str(urdf),'verification':asset_status},'contact_body_order':bodies,
        'contact_criterion':'net force norm >1N; max across original 3 physical samples (5ms apart, 10ms first-to-last); not load fraction',
        'reward_terms':{n:{'weight':env.reward_manager.get_term_cfg(n).weight,'function':str(env.reward_manager.get_term_cfg(n).func)} for n in env.reward_manager.active_terms},
        'reward_scaling':'read _step_reward (func*weight) times original .02 s; no recomputation',
        'video':'one rendered pre-reset frame each .02s, encoded 50fps; normal simulation-time speed' if args.video else None,
        'evaluation_lifecycle':'TRAIN original physics, policy history disabled in Isaac and maintained by E02 push_frame exactly once per returned live frame; pre-reset physical signals captured in reward hook; no training or statistic updates'})
    if startup_lock:
        fcntl.flock(startup_lock,fcntl.LOCK_UN);startup_lock.close();startup_lock=None
    write('frozen-before.json',{'module_fingerprint':before,'counts':counts,'normalizers':stats_before})
    capture={};old_reward=env.reward_manager.compute
    def reward_compute(dt):
        result=old_reward(dt);d=robot.data
        events=current_events(env)
        capture.clear();capture.update(serial({'effective_command':env.command_manager.get_command('base_velocity')[0],
            'pre_reset_body_linear_velocity':d.root_lin_vel_b[0],
            'root_linear_velocity_world':d.root_lin_vel_w[0],
            'all_joint_position':d.joint_pos[0,jointids],
            'all_joint_velocity':d.joint_vel[0,jointids],
            'all_joint_position_target':d.joint_pos_target[0,jointids],
            'all_joint_velocity_target':d.joint_vel_target[0,jointids],
            'pre_reset_velocity_supervision_scaled':d.root_lin_vel_b[0]*2.,
            'measured_body_vx_vy_wz':torch.cat((d.root_lin_vel_b[:,:2],d.root_ang_vel_b[:,2:3]),1)[0],
            'root_pos_w':d.root_pos_w[0],'heading_w':d.heading_w[0],'gravity_b':d.projected_gravity_b[0],
            'root_com_pos_w':d.root_com_pos_w[0],'root_quat_w':d.root_quat_w[0],
            'wheel_target_rad_s':d.joint_vel_target[0,[jointids[i] for i in wheel]],'wheel_actual_rad_s':d.joint_vel[0,[jointids[i] for i in wheel]],
            'leg_target_rad':d.joint_pos_target[0,[jointids[i] for i in leg]],'leg_actual_rad':d.joint_pos[0,[jointids[i] for i in leg]],
            'contact_force_history_max_N':sensor.data.net_forces_w_history[:,:,bodyids].norm(dim=-1).amax(1)[0],
            'contact_named_max_N':{k:sensor.data.net_forces_w_history[:,:,idx].norm(dim=-1).amax(1)[0] for k,idx in contact_indices.items()},
            'external_force_b_command':robot._external_force_b[0],'external_torque_b_command':robot._external_torque_b[0],
            'external_wrench_submission_enabled':bool(robot.has_external_wrench),
            'termination_terms':events,'reset_terminated':events['base_contact'],'reset_time_outs':events['time_out'],
            'reward':result[0],'reward_terms':dict(zip(env.reward_manager.active_terms,env.reward_manager._step_reward[0]*dt))}))
        if args.condition=='pulse':
            assert len(pulse_state['substeps'])==4
            capture['pulse_substeps']=list(pulse_state['substeps'])
            pulse_state['active']=False
        if args.video:capture['_image']=env.render().copy()
        return result
    env.reward_manager.compute=reward_compute
    records=[];start=time.perf_counter()
    with (out/'steps.jsonl').open('x',encoding='utf-8') as file,torch.inference_mode():
        for i in range(600 if args.condition=='standard' else 900):
            stage,cmd=schedule(i);effective=env.command_manager.get_command('base_velocity')
            assert torch.allclose(effective,torch.tensor([cmd],device=env.device),atol=1e-7,rtol=0)
            assert torch.allclose(history.history[:,-1,6:9],effective*torch.tensor([2.,2.,.25],device=env.device),atol=1e-6,rtol=0)
            if args.zero_reset_wrench and args.condition!='pulse':
                assert torch.count_nonzero(robot._external_force_b)==0 and torch.count_nonzero(robot._external_torque_b)==0
            features=features_from_history(encoder,normalizers,history.history)
            # Reuse the feature pass already needed by Actor: no extra policy sampling, forward, or observation compute.
            policy_command_before=history.history[0,-1,6:9].clone()
            raw_label_before=obs['critic'][0,:3].clone()
            true_velocity_before=robot.data.root_lin_vel_b[0].clone()
            world_velocity_before=robot.data.root_lin_vel_w[0].clone()
            assert torch.allclose(raw_label_before,true_velocity_before*2.,atol=1e-6,rtol=1e-5)
            estimator_diagnostic={'encoder_velocity_scaled':serial(features[0,:3]),
                'encoder_velocity_mps':serial(features[0,:3]/2.),
                'velocity_supervision_raw':serial(raw_label_before),'velocity_supervision_scale':2.,
                'velocity_true_mps_before':serial(true_velocity_before),
                'velocity_label_statistical_normalization':False,
                'policy_history_command_scaled_before':serial(policy_command_before),
                'policy_command_scale':[2.,2.,.25],
                'policy_normalizer_fingerprint_source':'frozen-before.json',
                'policy_normalizer_eps':normalizers.policy.eps}

            mu,log_std=actor.distribution_parameters(features)
            action=actor.deterministic_action(features) if args.execution=='D' else actor.sample_action_and_log_prob(features,generator=sample_generator)[0]
            assert torch.isfinite(action).all()
            com_before=robot.data.root_com_pos_w[0].clone();heading_before=robot.data.heading_w[0].clone()
            if args.condition=='pulse':
                pulse_state['substeps']=[]
                pulse_state['active']=True
            obs,reward,term,timeout,info=env.step(action)
            assert capture['reset_terminated']==bool(term[0]) and capture['reset_time_outs']==bool(timeout[0]), 'pre-reset done mismatch'
            if not (term[0] or timeout[0]):history.history,history.pushes=push_frame(history.history,history.pushes,obs['policy'])
            assert env._sim_step_counter==physics_start+(i+1)*cfg.decimation
            if args.zero_reset_wrench and args.condition!='pulse':
                assert not any(v for row in capture['external_force_b_command'] for v in row)
                assert not any(v for row in capture['external_torque_b_command'] for v in row)
            image=capture.pop('_image',None)
            capture['root_linear_acceleration_world_m_s2']=[(a-b)/.02 for a,b in zip(capture['root_linear_velocity_world'],serial(world_velocity_before))]
            row={**capture,**estimator_diagnostic,'episode_id':0,'step_index':i,'step':i,'stage':stage,'t_start_s':i*.02,'t_end_s':(i+1)*.02,'actor_action':serial(action[0]),'actor_mu':serial(mu[0]),'actor_std':serial(log_std[0].exp()),
                'root_com_pos_w_before':serial(com_before),'heading_w_before':serial(heading_before),'terminated':bool(term[0]),'time_out':bool(timeout[0])}
            file.write(json.dumps(row,allow_nan=False)+'\n');file.flush();records.append(row)
            if args.video:
                if image.size==0:raise RuntimeError('empty video frame')
                frame=cv2.cvtColor(image,cv2.COLOR_RGB2BGR);cv2.rectangle(frame,(0,0),(960,62),(0,0,0),-1)
                cv2.putText(frame,f"SAC {args.execution} seed{args.eval_seed} | sim t={(i+1)*.02:.2f}s",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(frame,f'{stage} | vx={cmd[0]:+.1f} m/s vy=0 wz=0',(12,51),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA);video.write(frame)
            if term[0] or timeout[0]:break
    if video:video.release();video=None
    if args.condition=='pulse':
        robot.set_external_force_and_torque(torch.zeros_like(robot._external_force_b),torch.zeros_like(robot._external_torque_b),is_global=False)
        assert torch.count_nonzero(robot._external_force_b)==0 and torch.count_nonzero(robot._external_torque_b)==0
        write('pulse-cleanup.json',{'zero_cache':True,'submission_enabled':bool(robot.has_external_wrench),'verified_physics_substeps':pulse_state['next_substep']})
    assert before==fingerprint(net) and counts=={n:m.count for n,m in net.named_modules() if hasattr(m,'count')}
    assert stats_before==serial(normalizers.export_statistics()),'normalizers changed during frozen evaluation'
    assert sha==sha256(args.checkpoint)
    write('frozen-after.json',{'module_fingerprint':fingerprint(net),'counts':{n:m.count for n,m in net.named_modules() if hasattr(m,'count')},'normalizers':serial(normalizers.export_statistics())})
    from r4_protocol import analyze_rows
    write('protocol-result.json',diagnostic_result(records,args.condition) if args.condition.startswith('diagnostic_') else analyze_rows(records,protocol=args.condition,pulse_direction=args.pulse_direction))
    metrics=([{'stage':k,**(v or {})} for k,v in diagnostic_result(records,args.condition)['windows'].items()] if args.condition.startswith('diagnostic_') else stage_metrics(records,'P00' if args.condition=='pulse' else args.condition))
    write('stage-metrics.json',metrics)
    import csv
    with (out/'stage-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in metrics for k in row)));writer.writeheader();writer.writerows(metrics)
    write('summary.json',{'status':'passed','execution':args.execution,'normalizers_unchanged':True,'development_gate_pass':d_gate(records,metrics) if args.condition=='standard' else None,'condition':args.condition,'plan_id':'SAC-R7P-PRECISION-PILOT-20261109','steps':len(records),'simulated_seconds':len(records)*.02,'evaluation_seed':args.eval_seed,
        'true_terminations':sum(r['terminated'] for r in records),'timeouts':sum(r['time_out'] for r in records),'elapsed_s':time.perf_counter()-start,
        'frozen_state_unchanged':True,'checkpoint_unchanged':True,'checkpoint_sha256':sha,'initial_state':initial,'all_finite':True,
        'physics_steps':env._sim_step_counter-physics_start,'physics_dt':env.physics_dt,'control_dt':env.step_dt,
        'rollout_finished_unix':time.time(),'stage_metrics':metrics,
        'stages':{name:sum(r['stage']==name for r in records) for name in sorted(set(r['stage'] for r in records))}})
    write('status.json',{'status':'passed'})
    progress.update('completed')
except BaseException as exc:
    write('error.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
    write('summary.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
    write('status.json',{'status':'failed','error':str(exc)})
    traceback.print_exc()
    progress.update('failed')
    raise
finally:
    if 'startup_lock' in globals() and startup_lock:startup_lock.close()
    if video:video.release()
    if env:
        if args.condition=='pulse' and 'robot' in globals():
            robot.set_external_force_and_torque(torch.zeros_like(robot._external_force_b),torch.zeros_like(robot._external_torque_b),is_global=False)
        env.close()
    if app:app.close()
