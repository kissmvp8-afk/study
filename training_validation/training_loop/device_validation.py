"""SAC-R1 设备针对性检查。人工环境检查不冒充真实批次/Isaac 验证。"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys
import tempfile
import torch
from runner import Trainer, to_learning_device, build_n_step_batch, features_from_history, compute_target
from config import cpu_debug_config
from fake_env import FakeLocomotionEnv


def validate_device_parity(trainer, source='fake'):
    """固定同一 CPU 批次与显式噪声，比较 CPU/CUDA 动作、密度与原 N05 多步目标。

    独立验证用抽样流，不推进训练的 collect/replay/learn。复制网络/统计，不修改训练状态。
    CUDA 不可用时仅做 CPU 自洽检查，报告清楚，不能当成 GPU 验证通过。
    """
    a=trainer.config.algorithm
    ids, slots=trainer.buffer.sample_starts(a.batch_size,a.n_steps,generator=torch.Generator().manual_seed(771))
    batch=build_n_step_batch(trainer.buffer.read_sequence(ids,slots,a.n_steps),a.gamma,a.n_steps)
    noise=torch.randn((a.batch_size,8),generator=torch.Generator().manual_seed(772))
    devices=['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu','cpu']
    results=[]
    for device in devices:
        encoder=copy.deepcopy(trainer.encoder).to(device).eval()
        actor=copy.deepcopy(trainer.actor).to(device).eval()
        targets=copy.deepcopy(trainer.targets).to(device).eval()
        normals=copy.deepcopy(trainer.normalizers)
        b=to_learning_device(batch,device)
        with torch.no_grad():
            features=features_from_history(encoder,normals,b.first.policy_history)
            action,logp=actor.sample_action_and_log_prob(features,noise=noise.to(device))
            live=b.bootstrap_mask.squeeze(1)>0
            class FixedNoiseActor:
                def sample_action_and_log_prob(self, features, generator=None):
                    return actor.sample_action_and_log_prob(features,noise=noise[live.cpu()].to(device))
            target=compute_target(b,encoder,FixedNoiseActor(),targets,normals,
                                  float(trainer.temperature.log_alpha.exp().detach().cpu()))
            assert torch.equal(target[~live],b.reward_sum[~live]), 'True termination bootstrapped'
            for value in (action,logp,target):
                assert bool(torch.isfinite(value).all())
            results.append([x.cpu() for x in (action,logp,target)])
    errors={}
    for name,x,y in zip(('action','log_prob','n_step_target'),*results):
        torch.testing.assert_close(x,y,rtol=3e-4,atol=3e-4)
        errors[name]=float((x-y).abs().max())
    return dict(status='passed',source=source,devices=devices,cuda_compared=devices[1]!='cpu',
        explicit_same_noise=True,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        max_abs_errors=errors,terminated_samples=int((batch.bootstrap_mask==0).sum()),
        timeout_samples=int(batch.tail.time_out.sum()),history_dtype=str(batch.first.policy_history.dtype),
        terminated_dtype=str(batch.first.terminated.dtype),index_dtype=str(batch.first.env_id.dtype),
        batch_size=a.batch_size,n_steps=a.n_steps)


def check_device(device):
    """真实更新、梯度隔离及带池保存恢复；同时测试两种回合结束分支。"""
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'checkpoints'))
    from checkpoint import save_checkpoint, restore_checkpoint, MODULE_NAMES
    cfg=replace(cpu_debug_config(),device=device)
    def new():
        return Trainer(cfg,FakeLocomotionEnv(3,6,11,[7,None,None]))
    trainer=new()
    trainer.collect_steps(24)
    before={n:{k:v.detach().clone() for k,v in getattr(trainer,n).state_dict().items()} for n in MODULE_NAMES}
    replay_before=trainer.generators['replay'].get_state().clone()
    report=validate_device_parity(trainer)
    assert torch.equal(replay_before,trainer.generators['replay'].get_state())
    trainer.learn_from_replay(2)
    def changed(name):
        return any(not torch.equal(v,getattr(trainer,name).state_dict()[k]) for k,v in before[name].items())
    assert all(changed(n) for n in ('actor','critics','targets','temperature'))
    assert not changed('encoder') and not changed('projector'), 'SAC gradient leaked into encoder'
    trainer.auxiliary_update()
    assert changed('encoder') and changed('projector')
    assert trainer.counters.critic_updates==2 and trainer.counters.actor_updates==1
    assert trainer.counters.alpha_updates==2 and trainer.counters.target_updates==2
    assert all(p.device.type==torch.device(device).type for n in MODULE_NAMES for p in getattr(trainer,n).parameters())
    assert trainer.buffer.device.type=='cpu' if hasattr(trainer.buffer,'device') else True
    with tempfile.TemporaryDirectory() as tmp:
        path=save_checkpoint(trainer,Path(tmp)/'restore.pt',include_replay=True)
        restored=new(); restored_report=restore_checkpoint(restored,path)
        assert restored_report.replay_restored
        for n in MODULE_NAMES:
            for k,v in getattr(trainer,n).state_dict().items():
                torch.testing.assert_close(v,getattr(restored,n).state_dict()[k],rtol=0,atol=0)
        assert asdict(trainer.counters)==asdict(restored.counters)
        for n in trainer.generators:
            assert torch.equal(trainer.generators[n].get_state(),restored.generators[n].get_state())
        assert trainer.normalizers.policy.count==restored.normalizers.policy.count
        assert trainer.normalizers.q.count==restored.normalizers.q.count
        trainer.learn_from_replay(2);restored.learn_from_replay(2)
        trainer.auxiliary_update();restored.auxiliary_update()
        for n in MODULE_NAMES:
            for k,v in getattr(trainer,n).state_dict().items():
                torch.testing.assert_close(v,getattr(restored,n).state_dict()[k],rtol=1e-6,atol=1e-7)
                if v.is_floating_point(): assert bool(torch.isfinite(v).all())
    return dict(status='passed',learner_device=device,parity=report,restore_and_next_update='passed',
                gradient_isolation='passed',counters=asdict(trainer.counters))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda:0'],required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    report=check_device(args.device)
    args.output_json.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))
