"""128环境夜间任务共用工具；不在导入时加载Torch，不改变SAC损失。"""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path

PARENT_SHA256='9552a0a94b1c50d6e4d78b85059f9d6f2ce671f28b2e7bc6589e5fe22d1e1e59'


def counts(trainer):
    """返回本任务计数快照；父Adam步数另行保存，不与本任务轮数混用。"""
    c=trainer.counters
    return asdict(c) if is_dataclass(c) else dict(vars(c))


@contextmanager
def frozen_replay_sampling(buffer,n_steps=5):
    """在不采集的学习窗口复用精确有效索引/连接关系；退出恢复方法，禁止窗口内写池。"""
    if getattr(buffer,'_night_sampling_frozen',False):
        raise RuntimeError('Nested frozen replay view is not allowed')
    names=('_continuity','valid_starts','append','load_state_dict')
    marker=object();saved={n:buffer.__dict__.get(n,marker) for n in names}
    old_continuity=buffer._continuity;old_valid=buffer.valid_starts
    writes=buffer._writes.clone()
    buffer._night_sampling_frozen=True
    try:
        continuity=old_continuity()
        buffer._continuity=lambda:continuity
        cache={n:old_valid(n) for n in sorted({1,n_steps})}
        def valid(n):
            if n not in cache:cache[n]=old_valid(n)
            return cache[n]
        def forbidden(*args,**kwargs):
            raise RuntimeError('Replay mutation inside frozen learning window')
        buffer.valid_starts=valid;buffer.append=forbidden;buffer.load_state_dict=forbidden
        yield
        if not buffer._writes.equal(writes):
            raise RuntimeError('Replay writes changed inside frozen window')
    finally:
        for name,value in saved.items():
            if value is marker:
                if name in buffer.__dict__:delattr(buffer,name)
            else:setattr(buffer,name,value)
        delattr(buffer,'_night_sampling_frozen')


def run_macro(trainer,use_cache=True):
    """采集8步×128env；预填充16宏轮后做8个(16Critic+16辅助)微循环。"""
    c=trainer.config.collection;a=trainer.config.algorithm
    assert (c.num_envs,c.steps_per_iteration,c.warmup_iterations,c.capacity_per_env)==(128,8,16,512)
    assert (a.critic_updates_per_iteration,a.auxiliary_updates_per_iteration,a.batch_size,a.n_steps)==(128,128,256,5)
    before=counts(trainer);learn_now=before['iterations']>=16
    log={'macro_iteration':before['iterations']+1,'collect':trainer.collect_steps(8),'microcycles':[]}
    if learn_now:
        context=frozen_replay_sampling(trainer.buffer,5) if use_cache else nullcontext()
        with context:
            for _ in range(8):
                learning=trainer.learn_from_replay(16)
                if learning.get('skipped',False):raise RuntimeError('Unexpected replay learning skip')
                auxiliary=[trainer.auxiliary_update() for _ in range(16)]
                log['microcycles'].append({'learn':learning,'auxiliary':auxiliary})
    trainer.counters.iterations+=1
    after=counts(trainer)
    expected={'iterations':1,'env_steps':8,'transitions':1024,'critic_updates':128 if learn_now else 0,
              'actor_updates':64 if learn_now else 0,'alpha_updates':128 if learn_now else 0,
              'target_updates':128 if learn_now else 0,'learning_skipped_iterations':0}
    for key,delta in expected.items():
        if after[key]-before[key]!=delta:raise RuntimeError('Macro counter mismatch: '+key)
    attempts=(after['auxiliary_updates']-before['auxiliary_updates']+
              after['auxiliary_skipped']-before['auxiliary_skipped'])
    if attempts!=(128 if learn_now else 0):raise RuntimeError('Auxiliary attempt count mismatch')
    def finite_log(value):
        if isinstance(value,float):return math.isfinite(value)
        if isinstance(value,dict):return all(finite_log(v) for v in value.values())
        if isinstance(value,list):return all(finite_log(v) for v in value)
        return True
    if not finite_log(log):raise FloatingPointError('Nonfinite macro log')
    log['counters']=after
    return log


def warm_start_r20(trainer,parent_path,alpha_multiplier,fork_seed):
    """16env父模型→128env新任务：继承模块/优化器/统计，新池、新计数、新随机流。"""
    import torch
    from checkpoint import _bounds_record
    path=Path(parent_path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PARENT_SHA256:
        raise ValueError('R20 parent model hash mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('format')!='tita-sac-checkpoint' or state.get('version')!=1:
        raise ValueError('Unsupported parent checkpoint')
    current=json.loads(json.dumps(asdict(trainer.config)));old=state['config']
    for key in ('task','network'):
        if json.loads(json.dumps(old[key]))!=current[key]:raise ValueError('Parent structure mismatch: '+key)
    for key,value in old['algorithm'].items():
        if key not in ('critic_updates_per_iteration','auxiliary_updates_per_iteration') and current['algorithm'][key]!=value:
            raise ValueError('Undeclared algorithm change: '+key)
    assert current['collection']==dict(num_envs=128,steps_per_iteration=8,warmup_iterations=16,capacity_per_env=512)
    assert current['algorithm']['critic_updates_per_iteration']==current['algorithm']['auxiliary_updates_per_iteration']==128
    assert current['seed']==fork_seed and fork_seed in (1001,1002)
    if state['action_bounds']!=_bounds_record(trainer) or state['contract_joint_names']!=list(trainer.contract.joint_names):
        raise ValueError('Action contract mismatch')
    if trainer.buffer.size or any(counts(trainer).values()):raise ValueError('Warm start requires empty new task')
    if alpha_multiplier not in (1.,.3,.1):raise ValueError('Undeclared alpha multiplier')
    for name in ('encoder','projector','actor','critics','targets','temperature'):
        getattr(trainer,name).load_state_dict(state['modules'][name],strict=True)
    for name,opt in trainer.optimizers.items():
        opt.load_state_dict(state['optimizers'][name])
        expected_lr=current['algorithm'][name+'_lr']
        if any(group['lr']!=expected_lr for group in opt.param_groups):raise ValueError('Optimizer LR mismatch')
    trainer.normalizers.load_statistics(state['normalizers'])
    trainer.encoder.eval()
    original_alpha=float(trainer.temperature.log_alpha.detach().exp().item())
    with torch.no_grad():trainer.temperature.log_alpha.add_(math.log(alpha_multiplier))
    effective_alpha=float(trainer.temperature.log_alpha.detach().exp().item())
    if abs(effective_alpha/(original_alpha*alpha_multiplier)-1)>1e-6:raise ValueError('Alpha intervention mismatch')
    for offset,name in enumerate(('collect','replay','learn'),1):trainer.generators[name].manual_seed(fork_seed+offset)
    return dict(mode='warm_start_modules_optimizers_normalizers_new_128env_replay',
                exact_resume=False,parent_sha256=PARENT_SHA256,parent_counters=state['counters'],
                inherited_optimizer_steps=True,task_counters_start_at_zero=True,parent_replay_imported=False,
                parent_episode_ids_imported=False,fork_seed=fork_seed,ancestor_training_seed=old['seed'],
                parent_alpha=original_alpha,alpha_multiplier=alpha_multiplier,effective_entry_alpha=effective_alpha,
                native_learned_std_restored=True)
