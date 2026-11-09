"""CPU只读核验真实下载的R8模型、优化器和两份完整恢复点；不恢复物理仿真。"""
import hashlib
import json
from pathlib import Path
import torch

B=Path('/workspace/sac-r8-review');v=json.loads((B/'download-inputs.json').read_text())
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def finite(x):
    n=0
    if isinstance(x,torch.Tensor):
        if x.is_floating_point():assert torch.isfinite(x).all()
        return 1
    if isinstance(x,dict):return sum(finite(y) for y in x.values())
    if isinstance(x,(tuple,list)):return sum(finite(y) for y in x)
    return n
def lr(s):return {n:[g['lr'] for g in x['param_groups']] for n,x in s['optimizers'].items()}
def opt_without_actor_lr(s):
    out={n:dict(x) for n,x in s.items()}
    out['actor']=dict(out['actor']);out['actor']['param_groups']=[dict(g) for g in out['actor']['param_groups']]
    for g in out['actor']['param_groups']:g.pop('lr')
    return out
def steps(s):
    result={}
    for k,opt in s['optimizers'].items():
        values=[float(z['step']) for z in opt['state'].values() if 'step' in z]
        result[k]=dict(min=min(values),max=max(values),state_count=len(values)) if values else None
    return result
def replay_boundary(s,name):
    r=s['replay'];st=r['storage'];seq=r['write_seq'];ids=torch.arange(128)
    old=(seq==143).nonzero();new=(seq==144).nonzero()
    assert old.shape==new.shape==(128,2) and torch.equal(old[:,0],ids) and torch.equal(new[:,0],ids)
    assert (r['writes']==512).all()
    assert (st['episode_id'][ids,new[:,1]]>st['episode_id'][ids,old[:,1]]).all()
    assert (st['step_index'][ids,new[:,1]]==0).all() and (st['history_length'][ids,new[:,1]]==1).all()
    history=st['policy_history'][ids,new[:,1]]
    assert torch.equal(history,history[:,-1:,:].expand_as(history))
    assert (torch.tensor(s['next_episode_ids'])>st['episode_id'].max(dim=1).values).all()
    return dict(name=name,envs_checked=128,writes_per_env=512,boundary_old_seq=143,boundary_new_seq=144,
        new_episode_ids_strictly_larger=True,first_new_step_zero=True,first_new_valid_history_length_one=True,
        first_history_padded_by_repeated_initial_frame=True,future_restore_episode_ids_above_replay_max=True,continuous_physics=False)
parent=torch.load('/workspace/sac-r7-plan/R1-1201-learning48-cbcd57a45c5addf7.pt',map_location='cpu',weights_only=True)
initial={};results=[];loaded={};boundary_results=[]
for a in v['L0_models']+v['L48_models']+v['paired1502_resumes']:
    p=B/a['name'];s=torch.load(p,map_location='cpu',weights_only=True)
    assert s['format']=='tita-sac-checkpoint' and s['version']==1
    expected=5e-5 if 'A0-' in p.name else 1e-5
    assert s['config']['algorithm']['actor_lr']==expected
    assert all(g['lr']==expected for g in s['optimizers']['actor']['param_groups'])
    for name,expected_other in dict(critic=2e-4,alpha=2e-5,auxiliary=.001).items():
        assert all(g['lr']==expected_other for g in s['optimizers'][name]['param_groups'])
    n=finite(s['modules']);assert n==finite(parent['modules'])
    finite(s['optimizers']);finite(s['normalizers'])
    record=dict(name=p.name,sha256=a['digest'][7:],bytes=p.stat().st_size,modules_tensors_checked=n,
        counters=s['counters'],actual_optimizer_lrs=lr(s),optimizer_steps=steps(s),
        alpha=float(s['modules']['temperature']['log_alpha'].exp()),
        top_keys=list(s),r8_metadata_keys=list(s.get('r8',{})),physical_simulation_restore_tested=False)
    if 'learning0' in p.name:
        assert equal(s['modules'],parent['modules'])
        assert equal(opt_without_actor_lr(s['optimizers']),opt_without_actor_lr(parent['optimizers']))
        assert all(s['counters'][k]==0 for k in ['critic_updates','actor_updates','alpha_updates','target_updates','auxiliary_updates'])
        assert s['counters']['iterations']==16 and s['counters']['transitions']==16384
        record['parent_modules_and_Adam_except_actor_lr_equal']=True
        record['L0_is_after_prefill_not_before_collection']=True
        record['normalizer_count_increase_during_prefill']={k:s['normalizers'][k]['count']-parent['normalizers'][k]['count'] for k in s['normalizers']}
        initial[p.name]=s
    else:
        for k,val in dict(iterations=64,env_steps=512,transitions=65536,critic_updates=6144,actor_updates=3072,alpha_updates=6144,target_updates=6144,auxiliary_updates=6144,auxiliary_skipped=0,learning_skipped_iterations=0).items():assert s['counters'][k]==val,(p.name,k,s['counters'][k])
        if 'fullresume' in p.name:
            assert s['replay'] is not None
            record['replay_keys']=list(s['replay'])
            record['replay_float_tensors_finite']=finite(s['replay'])
            model=next(x for x in v['L48_models'] if ('A0-1502' if 'A0-1502' in p.name else 'A1-1502') in x['name'])
            final=loaded[model['name']]
            for k in ('modules','optimizers','normalizers','counters','config'):assert equal(s[k],final[k]),(p.name,k)
            record['matches_L48_model_sections']=True
            boundary_results.append(replay_boundary(s,p.name))
    loaded[p.name]=s
    results.append(record)
for seed in (1501,1502):
    a=next(s for n,s in initial.items() if f'A0-{seed}' in n)
    b=next(s for n,s in initial.items() if f'A1-{seed}' in n)
    assert equal(a['normalizers'],b['normalizers'])
    assert equal(a['modules'],b['modules'])
    assert equal(opt_without_actor_lr(a['optimizers']),opt_without_actor_lr(b['optimizers']))
(B/'model-and-resume-verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
(B/'resume-boundary-audit.json').write_text(json.dumps(boundary_results,indent=2)+'\n')
print(json.dumps({'models_and_resumes_CPU_read':len(results),'L0_parent_inheritance_checks':4,'L48_counter_and_real_LR_checks':4,'full_replay_points_downloaded_and_read':2,'physics_resume_test':False}))
for a in v['paired1502_resumes']:
    s=loaded[a['name']]
    print(json.dumps({'name':a['name'],'replay':{k:({'tensor_shape':list(x.shape),'dtype':str(x.dtype)} if isinstance(x,torch.Tensor) else {'keys':list(x)} if isinstance(x,dict) else str(x)[:150]) for k,x in s['replay'].items()},'r8_metadata':{k:str(x)[:350] for k,x in s.get('r8',{}).items() if k!='global_rng'},'next_episode_ids':s.get('next_episode_ids',[])[:8]}))
