#!/usr/bin/env python3
"""R7P实际训练worker：固定父/奖励/计数，节点保存，无独立预检阶段。"""
import argparse,ast,copy,datetime as dt,fcntl,hashlib,importlib.util,inspect,json,os,random,shutil,signal,sys,time,traceback
from pathlib import Path
from dataclasses import asdict,is_dataclass
sys.dont_write_bytecode=True
PLAN="SAC-R7P-PRECISION-PILOT-20261109"
R1_SHA="46f4c8d0f3013ab95ca51db90babcbf9fcd53f259432abbf6cbf0e7a49153c9a"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp")
 with tmp.open("w") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def inside(root,value):
 p=(root/value).resolve();assert p.is_relative_to(root),p;return p
def timestamp(s):return dt.datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()

def rng_state():
 import torch,numpy as np
 n=np.random.get_state()
 return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
 import torch,numpy as np
 random.setstate(s["python"]);n=s["numpy"];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
 torch.set_rng_state(s["torch_cpu"]);torch.cuda.set_rng_state_all(s["torch_cuda"])

def make_hooks(extension):
 import torch
 from isaac_env import TitaSacHooks
 class R7PHooks(extension.R2Hooks):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw)
   manager=self.env.termination_manager
   assert set(manager.active_terms)=={"base_contact","time_out"}
   assert manager.get_term_cfg("base_contact").time_out is False
   assert manager.get_term_cfg("time_out").time_out is True
   self.current_events=None;self.event_calls=0;self.control_rows=[];self.input_snapshot=None
  def apply_action_and_simulate(self,action):
   if hasattr(self,"trainer"):
    obs=self.trainer.observation
    command=self.env.command_manager.get_command("base_velocity").detach().cpu().clone()
    history_command=obs.policy_history[:,-1,6:9].detach().cpu().clone()
    torch.testing.assert_close(history_command,command*torch.tensor([2.,2.,.25]),atol=1e-6,rtol=0)
    from runner import features_from_history
    with torch.no_grad():
     features=features_from_history(self.trainer.encoder,self.trainer.normalizers,obs.policy_history.to(self.trainer.device))
    lo,hi=self.trainer.config.task.velocity_label_columns
    self.input_snapshot=dict(predicted_velocity_scaled=features[:,:3].detach().cpu().tolist(),
     current_velocity_supervision_scaled=obs.q_observation[:,lo:hi].detach().cpu().tolist(),
     episode_id=obs.episode_id.tolist(),step_index=obs.step_index.tolist(),
     env_id=obs.env_id.tolist(),command=command.tolist(),history_command=history_command.tolist(),
     command_time_left=self.env.command_manager.get_term("base_velocity").time_left.detach().cpu().tolist(),
     action=action.detach().cpu().tolist())
   return super().apply_action_and_simulate(action)
  def compute_terminations(self):
   terminated,time_out=super().compute_terminations()
   self.current_events={"base_contact":terminated.detach().clone().cpu().bool(),"time_out":time_out.detach().clone().cpu().bool()}
   self.event_calls+=1
   return terminated,time_out
  def compute_rewards(self):
   self.assert_zero_wrench()
   result=TitaSacHooks.compute_rewards(self)
   from r7p_training import precision_values
   env=self.env;cmd=env.command_manager.get_command("base_velocity").detach().clone();d=self.robot.data
   mask,cost=precision_values(cmd,d.root_lin_vel_b,d.root_ang_vel_b)
   weight=-env.reward_manager.get_term_cfg("parking_precision").weight
   assert weight in (0.,.5) and abs(env.step_dt-.02)<1e-10
   extra=-weight*mask.to(cost.dtype)*cost*env.step_dt
   assert bool(torch.isfinite(cost).all()) and bool(torch.isfinite(result).all())
   assert bool((extra.abs()<=.01250001).all())
   index=env.reward_manager.active_terms.index("parking_precision")
   terms=env.reward_manager._step_reward.detach()*env.step_dt
   torch.testing.assert_close(terms[:,index],extra,rtol=1e-5,atol=1e-7)
   torch.testing.assert_close(result,terms.sum(1),rtol=1e-5,atol=1e-6)
   original=terms[:,:index].sum(1)+terms[:,index+1:].sum(1)
   torch.testing.assert_close(result,original+extra,rtol=1e-5,atol=1e-6)
   if weight==0:torch.testing.assert_close(result,original,rtol=0,atol=1e-6)
   self.precision_snapshot=dict(mask=mask.cpu().tolist(),cost=cost.cpu().tolist(),weight=weight,dt=env.step_dt,extra=extra.cpu().tolist(),original_reward=original.cpu().tolist(),total_reward=result.detach().cpu().tolist(),manager_single_compute_verified=True)
   assert self.current_events is not None
   env=self.env;command=env.command_manager.get_command("base_velocity").detach().clone()
   raw=extension.parking_body_velocity(env).detach();weighted=raw*self.weight*env.step_dt
   index=env.reward_manager.active_terms.index("parking_body_velocity")
   torch.testing.assert_close(env.reward_manager._step_reward[:,index].detach()*env.step_dt,weighted)
   if hasattr(self,"r7p_output") and not getattr(self,"first_reward_verified",False):
    write(self.r7p_output/"first-step-reward-check.json",dict(status="passed_runtime",event_calls=self.event_calls,precision=self.precision_snapshot,old_parking_weight=self.weight,command_resampling_time_range=list(env.cfg.commands.base_velocity.resampling_time_range)))
    self.first_reward_verified=True
   wheels=self.robot.data.joint_vel_target[:,self.wheel_ids].detach()
   for value in (command,raw,weighted,wheels):assert bool(torch.isfinite(value).all())
   assert not bool((wheels.abs()>20.0001).any())
   self.samples.append(dict(commands=command.cpu(),raw=raw.cpu(),weighted=weighted.cpu(),wheels=wheels.cpu(),terms=self.current_events))
   if self.input_snapshot is not None:
    row=dict(self.input_snapshot,precision=self.precision_snapshot,terminated=self.current_events["base_contact"].tolist(),
     timeout=self.current_events["time_out"].tolist(),pre_reset_command=command.cpu().tolist(),
     parking_raw=raw.cpu().tolist(),parking_weighted_dt=weighted.cpu().tolist(),
     robot_joint_names=list(self.robot.joint_names),
     all_joint_position=self.robot.data.joint_pos.detach().cpu().tolist(),
     all_joint_velocity=self.robot.data.joint_vel.detach().cpu().tolist(),
     all_joint_position_target=self.robot.data.joint_pos_target.detach().cpu().tolist(),
     all_joint_velocity_target=self.robot.data.joint_vel_target.detach().cpu().tolist(),
     root_pos_w=self.robot.data.root_pos_w.detach().cpu().tolist(),
     contact_body_names=list(env.scene['contact_forces'].body_names),
     contact_net_forces_w_history=env.scene['contact_forces'].data.net_forces_w_history.detach().cpu().tolist(),
     pre_reset_velocity_supervision_scaled=(2*self.robot.data.root_lin_vel_b).detach().cpu().tolist(),
     vx_vy_wz=torch.cat((self.robot.data.root_lin_vel_b[:,:2],self.robot.data.root_ang_vel_b[:,2:3]),1).detach().cpu().tolist())
    self.control_rows.append(row);self.input_snapshot=None
   self.current_events=None
   return result
  def drain(self):
   result=super().drain();result["scope"]="8 steps x 128 environments, pre-reset"
   result["termination_event_source"]="current_compute_return"
   result["per_environment_control_steps"]=self.control_rows
   result["reward_scope"]="parking_raw and parking_weighted_dt are current step; R1 episode reward logs are latest reset cumulative"
   self.control_rows=[]
   return result
 return R7PHooks

def effective(trainer,hooks,parameters,direction,entry):
 import torch,runner
 c=trainer.config.collection;a=trainer.config.algorithm;env=hooks.env
 assert runner.current_alpha_tensor(trainer.temperature).device==trainer.temperature.log_alpha.device
 actual=dict(num_envs=env.num_envs,steps_per_macro=c.steps_per_iteration,warmup_macros=c.warmup_iterations,
 capacity_per_env=c.capacity_per_env,replay_capacity_total=trainer.buffer.num_envs*trainer.buffer.capacity_per_env,
 critic_updates_per_learning_macro=a.critic_updates_per_iteration,actor_updates_per_learning_macro=a.critic_updates_per_iteration//a.actor_update_every,
 alpha_updates_per_learning_macro=a.critic_updates_per_iteration//a.alpha_update_every,target_updates_per_learning_macro=a.critic_updates_per_iteration//a.target_update_every,
 auxiliary_attempts_per_learning_macro=a.auxiliary_updates_per_iteration,microcycles=8,critic_updates_per_microcycle=16,auxiliary_attempts_per_microcycle=a.auxiliary_updates_per_iteration//8,freeze_representation=direction["freeze_representation"],
 batch_size=a.batch_size,auxiliary_batch_size=a.auxiliary_batch_size,gamma=a.gamma,n_steps=a.n_steps,tau=a.tau,
 standing_command_fraction=env.cfg.commands.base_velocity.rel_standing_envs,
 policy_observation_noise="on" if env.cfg.observations.policy.enable_corruption else "off",
 reset_force_N=list(env.cfg.events.base_external_force_torque.params["force_range"]),
 reset_torque_Nm=list(env.cfg.events.base_external_force_torque.params["torque_range"]),
 interval_push="off" if env.cfg.events.push_robot is None else "on",
 physics_dt_s=float(env.physics_dt),decimation=int(env.cfg.decimation),control_dt_s=float(env.step_dt),
 tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
 alpha_compute="GPU_tensor_common_to_all",
 normalizers="frozen_bundle" if direction["freeze_representation"] else "inherit_then_update_during_training_freeze_during_eval",
 network_reward_action_bounds_timeout_semantics="R4_except_declared_parking_weight_command_duration_and_logging",
 initial_std="restore_learned_value_do_not_reset_or_scale_for_evaluation",
 alpha_entry_multiplier=entry["alpha_multiplier"],
 track_ang_vel_z_exp_weight=env.reward_manager.get_term_cfg("track_ang_vel_z_exp").weight,
 parking_body_velocity_weight=env.reward_manager.get_term_cfg("parking_body_velocity").weight,
 command_resampling_time_range=list(env.cfg.commands.base_velocity.resampling_time_range),new_parking_precision_weight=-env.reward_manager.get_term_cfg("parking_precision").weight)
 for key in ("critic","actor","alpha","auxiliary"):
  values={g["lr"] for g in trainer.optimizers[key].param_groups};assert len(values)==1
  actual[key+"_lr"]=values.pop()
 from r7p_training import EFFECTIVE_COMMON
 for key,value in EFFECTIVE_COMMON.items():assert actual[key]==value,(key,actual[key],value)
 for key,value in direction.items():
  if key!="id":assert actual[key]==value,(key,actual[key],value)
 return actual

def main():
 begin=time.time();p=argparse.ArgumentParser()
 p.add_argument("--worker-config",type=Path,required=True);p.add_argument("--request",type=Path,required=True);p.add_argument("--response",type=Path,required=True)
 import tensordict,h5py
 from isaaclab.app import AppLauncher
 AppLauncher.add_app_launcher_args(p);args=p.parse_args()
 cfg=json.loads(args.worker_config.read_text());request=json.loads(args.request.read_text())
 assert cfg["use_cache"] is False,"R6 replay cache must remain disabled"
 root=Path(request["workroot"]).resolve();assert request["plan_id"]==PLAN
 parameters_path=inside(root,request["parameters_path"]);parameters=json.loads(parameters_path.read_text());assert parameters["plan_id"]==PLAN
 from r7p_training import validate_plan
 validate_plan(parameters)
 deadline=timestamp(request["deadline_utc"]);assert time.time()<deadline-60
 preflight=False
 assert request["action"]=="train_evaluate"
 assert not request.get("calibration_recovery",False),"R6 smoke cannot be resumed into a formal lane"
 assert preflight or request["action"]=="train_evaluate"
 from r7p_training import selection_record,inherited_direction,configure_arm,barrier_wait,parking_precision_cost
 selection_path=inside(root,request["parent_selection_path"]);selection=selection_record(selection_path,root)
 direction=inherited_direction(selection,request["direction"])
 assert request["execution_mode"]=="four_lane"
 seed=request["fork_seed"];assert seed in (1401,1402)
 target=request.get("preflight_learning_macros",2) if preflight else request["target_learning_macros"];assert isinstance(target,int) and target>=1
 assert (2<=target<=80) if preflight else target==48
 output=inside(root,"preflight/"+direction["id"]+"-"+str(seed) if preflight else "arms/"+direction["id"]+"-"+str(seed));output.mkdir(parents=True,exist_ok=True)
 source_root=Path(cfg["study_root"]).resolve();args.study_root=source_root
 args.r1_trainer=Path(cfg["r1_trainer"]);args.r2_tools=Path(cfg["r2_tools"]);args.output_dir=output
 assert sha(args.r1_trainer)==R1_SHA
 assert not root.is_relative_to(source_root)
 args.root=Path(cfg["root"]);args.arm="R20";args.stage="R7P";args.learner_device="cuda:0";args._env=None;args.seed=seed
 sys.path.insert(0,str(args.r2_tools));sys.path.insert(0,str(args.r1_trainer.parent))
 from linux_common import prepare_args
 args=prepare_args(args)
 stopped={"value":False}
 def graceful_stop_handler(signum,frame):
  stopped.update(value=True,signal=int(signum))
 lock=Path(cfg["lock_file"]).open("a");fcntl.flock(lock,fcntl.LOCK_EX)
 app=None;trainer=None;last_valid=None
 status=dict(status="starting",request_sha256=sha(args.request),direction=direction["id"],fork_seed=seed)
 write(output/"STATUS.json",status)
 try:
  import torch
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  base=module("r6_original_r1",args.r1_trainer);base.install_paths(source_root)
  app=AppLauncher(args).app
  # AppLauncher installs its own abort handlers; override only after construction.
  signal.signal(signal.SIGTERM,graceful_stop_handler)
  signal.signal(signal.SIGINT,graceful_stop_handler)
  assert signal.getsignal(signal.SIGTERM) is graceful_stop_handler
  assert signal.getsignal(signal.SIGINT) is graceful_stop_handler
  import r2_extension as extension
  from checkpoint import MODULE_NAMES,restore_checkpoint,collect_training_state
  from r7p_counts import counts
  from r7p_training import run_macro,warm_start_r7p,assert_r7p_counts
  hooks_class=make_hooks(extension)
  def configure(env_cfg,_arm):
   env_cfg=extension.configure(env_cfg,"R20")
   env_cfg.rewards.parking_body_velocity.weight=direction["parking_body_velocity_weight"]
   env_cfg.commands.base_velocity.resampling_time_range=tuple(direction["command_resampling_time_range"])
   env_cfg.rewards.track_ang_vel_z_exp.weight=direction["track_ang_vel_z_exp_weight"]
   from isaaclab.managers import RewardTermCfg
   assert not hasattr(env_cfg.rewards,"parking_precision")
   env_cfg.rewards.parking_precision=RewardTermCfg(func=parking_precision_cost,weight=-direction["new_parking_precision_weight"])
   return env_cfg
  source=inspect.getsource(base.make_trainer)
  replacements=[
   ("make_sac_cfg('isaac_flat', 16, 0, args.device or 'cuda:0')","make_sac_cfg('isaac_flat', 128, args.seed, args.device or 'cuda:0')"),
   ("cfg.commands.base_velocity.rel_standing_envs = .02 if args.arm == 'S0' else .20","cfg = configure(cfg, 'R20')"),
   ("required, seed=0,","required, seed=args.seed,"),
   ("capacity_per_env=4096","capacity_per_env=512"),
   ("warmup_iterations=128","warmup_iterations=16"),
   ("critic_updates_per_iteration=16","critic_updates_per_iteration=128"),
   ("auxiliary_updates_per_iteration=16","auxiliary_updates_per_iteration=128"),
   ("hooks = TitaSacHooks(env, False)","hooks = R7PHooks(env, False)")]
  for old,new in replacements:
   assert source.count(old)==1,old;source=source.replace(old,new,1)
  ast.parse(source);namespace=dict(base.__dict__);namespace.update(PLAN_ID=PLAN,configure=configure,R7PHooks=hooks_class)
  exec(compile(source,"<R6-original-Trainer-constructor>","exec"),namespace)
  (output/"effective_make_trainer.py").write_text(source)
  trainer,hooks=namespace["make_trainer"](args)
  configure_arm(trainer,direction["id"])
  hooks.trainer=trainer;hooks.r7p_output=output
  # Environment construction may install callbacks; enforce our handler again.
  signal.signal(signal.SIGTERM,graceful_stop_handler)
  signal.signal(signal.SIGINT,graceful_stop_handler)
  assert signal.getsignal(signal.SIGTERM) is graceful_stop_handler
  assert signal.getsignal(signal.SIGINT) is graceful_stop_handler
  status["startup_seconds"]=time.time()-begin
  import resource
  collect_times=[]
  original_collect=trainer.collect_steps
  def timed_collect(*a,**kw):
   torch.cuda.synchronize();started=time.perf_counter()
   result=original_collect(*a,**kw);torch.cuda.synchronize()
   collect_times.append(time.perf_counter()-started);return result
  trainer.collect_steps=timed_collect
  protected=list(source_root.rglob("*.py"))+[args.r1_trainer,Path(extension.__file__),Path(__file__),Path(__file__).with_name("r7p_counts.py"),Path(__file__).with_name("r7p_training.py"),selection_path,Path(selection["resolved_parent_effective_settings_path"]),parameters_path,args.worker_config]
  hashes={str(x.resolve()):sha(x) for x in protected};write(output/"source_hashes.json",hashes)
  def verify_sources():
   for path,value in hashes.items():assert sha(path)==value,path
  identity=dict(plan_id=PLAN,direction=direction["id"],fork_seed=seed,parameters_sha256=sha(parameters_path),parent_selection_sha256=sha(selection_path),execution_mode=request["execution_mode"])
  resume=output/"latest-fullresume.pt";resume_record=output/"latest-fullresume.json"
  parent=Path(selection["resolved_model_path"]);assert parent.stat().st_size==selection["selected"]["bytes"]
  assert sha(parent)==selection["selected"]["model_sha256"]
  if resume.exists():
   assert not preflight,"R6 smoke is disposable and must start from the parent"
   record=json.loads(resume_record.read_text());assert record["sha256"]==sha(resume)
   state=torch.load(resume,map_location="cpu",weights_only=True);assert state["r7p"]["identity"]==identity
   assert state["replay"] is not None and state["replay"]["num_envs"]==128
   report=restore_checkpoint(trainer,resume);assert report.mode=="exact_replay" and report.replay_restored
   restore_rng(state["r7p"]["global_rng"]);entry=state["r7p"]["entry"]
   assert entry["parent_sha256"]==selection["selected"]["model_sha256"]
   for key,opt in trainer.optimizers.items():
    assert all(g["lr"]==getattr(trainer.config.algorithm,key+"_lr") for g in opt.param_groups)
   assert trainer.observation.episode_id.tolist()==state["next_episode_ids"]
   use_cache=state["r7p"]["use_cache"];assert use_cache==cfg["use_cache"]
   write(output/("resume-"+str(time.time_ns())+".json"),dict(report=asdict(report),sha256=sha(resume),identity=identity))
  else:
   assert not request.get("resume_latest_128env",False),"strict resume requested but no full checkpoint"
   entry=warm_start_r7p(trainer,parent,seed,selection,direction["id"],base.fingerprint)
   import numpy as np
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   use_cache=False if preflight else cfg["use_cache"]
   write(output/"warm-start-lineage.json",entry)
  frozen=None
  initial_physical={name:getattr(hooks.robot.data,name).detach().cpu().clone() for name in
   ("root_pos_w","root_quat_w","root_lin_vel_b","root_ang_vel_b","joint_pos","joint_vel")}
  physical_record=dict(sha256=base.fingerprint(initial_physical),fork_seed=seed,scope="initial untrained body/joint state; command excluded by intervention")
  if not resume.exists() and not preflight:
   pairing=root/"paired-starts";pairing.mkdir(exist_ok=True);reference=pairing/(str(seed)+".json")
   if reference.exists():assert json.loads(reference.read_text())["sha256"]==physical_record["sha256"],"unmatched same-seed initial physical state"
   else:write(reference,physical_record)
  write(output/("initial-physical-"+str(time.time_ns())+".json"),physical_record)
  settings=effective(trainer,hooks,parameters,direction,entry);write(output/"effective-settings.json",settings)
  with torch.no_grad():
   from runner import features_from_history
   features=features_from_history(trainer.encoder,trainer.normalizers,trainer.observation.policy_history.to(trainer.device))
   _,std_log=trainer.actor.distribution_parameters(features)
   write(output/("entry-exploration-"+str(time.time_ns())+".json"),dict(alpha=float(trainer.temperature.log_alpha.detach().exp()),std_mean=float(std_log.exp().mean()),std_min=float(std_log.exp().min()),std_max=float(std_log.exp().max()),restored_not_reinitialized=True))
  initial={n:base.fingerprint(getattr(trainer,n).state_dict()) for n in MODULE_NAMES}
  write(output/("entry-fingerprints-"+str(time.time_ns())+".json"),dict(modules=initial,optimizers=base.fingerprint({n:o.state_dict() for n,o in trainer.optimizers.items()}),normalizers=base.fingerprint(trainer.normalizers.export_statistics()),entry=entry))
  if not resume.exists():
   learned_start=dict(modules=initial,optimizers=base.fingerprint({n:o.state_dict() for n,o in trainer.optimizers.items()}),normalizers=base.fingerprint(trainer.normalizers.export_statistics()))
   pairdir=root/"paired-learning-starts";pairdir.mkdir(exist_ok=True);pairpath=pairdir/(str(seed)+".json")
   if pairpath.exists():assert json.loads(pairpath.read_text())==learned_start,"same-fork parent learning states differ"
   else:write(pairpath,learned_start)
  import runner
  step_counts=dict(state["r7p"]["actual_update_counts"]) if resume.exists() else {k:0 for k in (*trainer.optimizers,"target")}
  old_target=runner.soft_update
  def counted_target(*a,**kw):
   result=old_target(*a,**kw);step_counts["target"]+=1;return result
  runner.soft_update=counted_target
  for key,opt in trainer.optimizers.items():
   old=opt.step
   def step(*a,_key=key,_old=old,**kw):
    result=_old(*a,**kw);step_counts[_key]+=1;return result
   opt.step=step
  def save(name,replay):
   verify_sources();base.check_finite(trainer)
   if frozen:frozen.check()
   c=counts(trainer)
   for key,counter in (("critic","critic_updates"),("actor","actor_updates"),("alpha","alpha_updates"),("target","target_updates"),("auxiliary","auxiliary_updates")):
    assert step_counts[key]==c[counter],("actual_cumulative_steps",key)
   state=collect_training_state(trainer,replay)
   state["r7p"]=dict(identity=identity,entry=entry,use_cache=use_cache,global_rng=rng_state(),frozen_bundle=frozen.check() if frozen else None,actual_update_counts=dict(step_counts))
   path=output/name;tmp=path.with_suffix(".tmp")
   with tmp.open("wb") as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
   checked=torch.load(tmp,map_location="cpu",weights_only=True)
   assert checked["counters"]==counts(trainer) and (checked["replay"] is not None)==replay
   assert base.fingerprint(checked)==base.fingerprint(state),"checkpoint readback mismatch"
   os.replace(tmp,path)
   result=dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size,counters=counts(trainer),readback_verified=True)
   if replay and name=="latest-fullresume.pt":write(resume_record,result)
   return result
  last_save=time.monotonic();timings=[];cache_report={"use_cache":False,"status":"disabled_by_R6_plan"};last_valid=counts(trainer)
  barrier_done=False
  # Serialize environment construction and prefill, then release initialization lock.
  def release_initialization():fcntl.flock(lock,fcntl.LOCK_UN)
  assert_r7p_counts(trainer)
  with (output/"macros.jsonl").open("a") as logs:
   while max(0,trainer.counters.iterations-16)<target:
    if stopped["value"] or time.time()>=deadline-60 or shutil.disk_usage(root).free<10*2**30:
     status.update(status="budget_stopped",checkpoint=save("latest-fullresume.pt",True));write(output/"STATUS.json",status);return 2
    if not barrier_done and trainer.counters.iterations>=16:
     save("latest-fullresume.pt",True)
     release_initialization()
     try:
      barrier_wait(root,request,identity,counts(trainer),write,lambda:stopped["value"] or time.time()>=deadline-60 or shutil.disk_usage(root).free<10*2**30)
     except TimeoutError as exc:
      status.update(status="barrier_stopped",error=str(exc),checkpoint=save("latest-fullresume.pt",True));write(output/"STATUS.json",status);return 2
     barrier_done=True
    verify_sources()
    if preflight and trainer.counters.iterations==16:
     cache_report={"use_cache":False,"status":"disabled_by_R6_plan"};use_cache=False;write(output/"cache-equivalence.json",cache_report)
     # Probe updates are discarded; counters/optimizer hooks restart only their diagnostic increments.
     step_counts.update({k:0 for k in step_counts})
    before=counts(trainer);before_steps=dict(step_counts);calls=hooks.event_calls
    torch.cuda.synchronize();started=time.perf_counter()
    log=run_macro(trainer,use_cache=use_cache)
    log["diagnostics"]=hooks.drain()
    base.check_finite(trainer)
    if frozen:log["frozen_bundle_assertion"]=frozen.check()
    assert_r7p_counts(trainer)
    delta={k:step_counts[k]-before_steps[k] for k in step_counts};c=counts(trainer);learning=before["iterations"]>=16
    assert delta["critic"]==delta["alpha"]==delta["target"]==(128 if learning else 0)
    assert delta["actor"]==(64 if learning else 0)
    assert delta["auxiliary"]==c["auxiliary_updates"]-before["auxiliary_updates"]
    assert hooks.event_calls-calls==8
    log["optimizer_steps_actual"]=delta
    log["target_updates_actual"]=delta["target"]
    log["normalizer_counts"]={name:getattr(trainer.normalizers,name).count for name in ("policy","q")}
    logs.write(json.dumps(log,allow_nan=False)+"\n");logs.flush()
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    timing=dict(iteration=c["iterations"],learning=learning,wall_s=elapsed,epoch=time.time(),
     max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
     gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    timings.append(timing)
    with (output/"timings.jsonl").open("a") as f:f.write(json.dumps(timing)+"\n");f.flush()
    attempts=c["auxiliary_updates"]+c["auxiliary_skipped"]
    assert attempts==0 or c["auxiliary_skipped"]/attempts<=.01,"auxiliary skip ratio exceeds1%"
    if c["iterations"]-16 in (24,48):
     level=c["iterations"]-16
     node=dict(model=save("model-learning-%d.pt"%level,False),full=save("fullresume-learning-%d.pt"%level,True))
     write(output/("node-learning-%d.json"%level),node)
     save("latest-fullresume.pt",True);last_save=time.monotonic()
    last_valid=c
    status.update(status="running",counters=c,heartbeat_epoch=time.time(),elapsed_s=time.time()-begin)
    write(output/"STATUS.json",status)
    if time.monotonic()-last_save>=300:
     save("latest-fullresume.pt",True);last_save=time.monotonic()
  full=save("latest-fullresume.pt",True)
  model=save("model-learning-48.pt",False)
  assert sha(parent)==selection["selected"]["model_sha256"]
  result=dict(status="trained",training_arm=dict(direction=direction["id"],fork_seed=seed,learning_macros=48,
    counters=counts(trainer),model_path=model["path"],model_sha256=model["sha256"],
    settings_verified=True,effective_settings_path=str((output/"effective-settings.json").relative_to(root)),cases=[]),checkpoint=full,use_cache=False)
  status.update(status="completed",counters=counts(trainer));write(output/"STATUS.json",status);write(args.response,result);return 0
 except BaseException as exc:
  status.update(status="failed",error=repr(exc),traceback=traceback.format_exc(),last_valid_counters=last_valid,
   last_checkpoint_retained=str(output/"latest-fullresume.pt"),fault_state_not_promoted=True)
  write(output/"STATUS.json",status);return 1
 finally:
  if args._env is not None:args._env.close()
  if app is not None:app.close()
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=="__main__":sys.exit(main())

