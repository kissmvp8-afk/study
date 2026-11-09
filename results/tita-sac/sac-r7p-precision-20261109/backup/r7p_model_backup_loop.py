"""R7P固定4臂模型备份；CPU受限读取、稳定硬链接、30分钟增量和明确覆盖索引。"""
import argparse
import os
from pathlib import Path
import shutil
import time
from r7p_backup_ops import PLAN,atomic,read,sha
from r7p_publish import publish,REPO,BRANCH,TAG
ARMS=('P0','P1');FORKS=(1401,1402);STAGES=(24,48)

def expected_files(root):
    for arm in ARMS:
        for fork in FORKS:
            for stage in STAGES:
                yield dict(direction=arm,fork_seed=fork,stage=stage,kind='model',source=Path(root)/'arms'/('%s-%d'%(arm,fork))/('model-learning-%d.pt'%stage))
                yield dict(direction=arm,fork_seed=fork,stage=stage,kind='resume',source=Path(root)/'arms'/('%s-%d'%(arm,fork))/('fullresume-learning-%d.pt'%stage))
            yield dict(direction=arm,fork_seed=fork,stage=None,kind='resume',source=Path(root)/'arms'/('%s-%d'%(arm,fork))/'latest-fullresume.pt')

def validate_metadata(state,item,parameters_sha,parent_sha,mode):
    if state.get('format')!='tita-sac-checkpoint' or state.get('version')!=1:raise ValueError('Checkpoint format mismatch')
    if set(state['modules'])!={'encoder','projector','actor','critics','targets','temperature'}:raise ValueError('Incomplete modules')
    identity=state['r7p']['identity']
    expected=dict(plan_id=PLAN,direction=item['direction'],fork_seed=item['fork_seed'],parameters_sha256=parameters_sha,parent_selection_sha256=parent_sha,execution_mode=mode)
    if identity!=expected:raise ValueError('Training identity mismatch')
    counts=state['counters'];learning=counts['iterations']-16
    if learning<0 or learning>48 or (item['stage'] is not None and learning!=item['stage']):raise ValueError('Unexpected learning stage')
    for k,v in dict(env_steps=8*(learning+16),transitions=1024*(learning+16),critic_updates=128*learning,alpha_updates=128*learning,target_updates=128*learning,actor_updates=64*learning).items():
        if counts[k]!=v:raise ValueError('Checkpoint counter mismatch '+k)
    expected_aux=128*learning
    if counts['auxiliary_updates']+counts['auxiliary_skipped']!=expected_aux:raise ValueError('Auxiliary count mismatch')
    actual=state['r7p']['actual_update_counts']
    for key,counter in (('critic','critic_updates'),('actor','actor_updates'),('alpha','alpha_updates'),('target','target_updates'),('auxiliary','auxiliary_updates')):
        if actual[key]!=counts[counter]:raise ValueError('Actual optimizer count mismatch')
    if (state['replay'] is not None)!=(item['kind']=='resume'):raise ValueError('Replay kind mismatch')
    if state['r7p'].get('use_cache') is not False:raise ValueError('Unexpected replay cache')
    if state['r7p']['entry']['parent_sha256'] is None:raise ValueError('Missing source parent')
    return dict(identity=identity,counters=counts,actual_update_counts=actual,parent_sha256=state['r7p']['entry']['parent_sha256'],learning_macros=learning)

def load_checkpoint(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True,mmap=True)

def fingerprint_stat(p):
    st=p.stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)

def remove_redundant_link(staged,source):
    """仅源路径仍指向同一inode时删除多余硬链接；从不删除唯一副本。"""
    if staged.exists() and source.exists() and os.path.samefile(staged,source) and staged.stat().st_nlink>=2:
        staged.unlink();return True
    return False

def tick(config,loader=load_checkpoint):
    root=Path(config['workroot']).resolve();out=Path(config['model_backup_root']).resolve();out.mkdir(parents=True,exist_ok=True)
    staging=out/'staging';staging.mkdir(exist_ok=True)
    if out.stat().st_dev!=root.stat().st_dev:raise ValueError('Model backup must share filesystem; no large fallback copies')
    if not (out/'pending.sidecar.json').exists() and not any(i['source'].is_file() for i in expected_files(root)):
        report=dict(epoch=time.time(),uploaded=[],skipped=[],status='no_formal_models_yet')
        atomic(out/('record-%d.sidecar.json'%time.time_ns()),report);return report
    parameters=root/config['parameters_path'];parent=root/'parent-selection.json'
    param_sha=sha(parameters);parent_sha=sha(parent);parent_record=read(parent)
    if config.get('parameters_sha256')!=param_sha or config.get('parent_selection_sha256')!=parent_sha:raise ValueError('Frozen configuration digest mismatch')
    index_path=out/'coverage.sidecar.json';index=read(index_path) if index_path.exists() else dict(assets={})
    registry_path=out/'staging-registry.sidecar.json'
    registry=read(registry_path) if registry_path.exists() else {}
    report=dict(epoch=time.time(),uploaded=[],skipped=[],retained_unique_staged=[])
    pending_path=out/'pending.sidecar.json'
    def finish(pending):
        # r7p_publish reconciles existing ref/release assets before any mutation.
        result=publish(pending['plan_path'],config['gh'],out/'receipts')
        index['assets'][pending['sha256']]=dict(pending,remote_status=result['status'],verified_epoch=time.time())
        atomic(index_path,index)
        if not remove_redundant_link(Path(pending['staged']),Path(pending['source'])):
            report['retained_unique_staged'].append(pending['staged'])
        pending_path.unlink();report['uploaded'].append(pending['sha256'])
    if pending_path.exists():finish(read(pending_path))
    completed=0
    for item in expected_files(root):
        if completed>=config.get('max_assets_per_tick',8) or time.time()>=config.get('until_epoch',float('inf')):break
        source=item['source']
        if not source.is_file():continue
        if source.is_symlink():raise ValueError('Model symlink not permitted')
        before=fingerprint_stat(source);record_path=source.with_name('node-learning-%d.json'%item['stage'] if item['kind']=='resume' and item['stage'] is not None else 'latest-fullresume.json')
        record_bytes=None
        if item['kind']=='resume':
            if not record_path.is_file():report['skipped'].append(dict(source=str(source),reason='resume_sidecar_pending'));continue
            record_bytes=record_path.read_bytes();record=__import__('json').loads(record_bytes)
            if item['stage'] is not None:record=record['full']
            if record.get('readback_verified') is not True or record.get('bytes')!=before[2] or record.get('path')!=str(source.relative_to(root)):
                report['skipped'].append(dict(source=str(source),reason='resume_sidecar_not_yet_matching'));continue
        digest=sha(source)
        if fingerprint_stat(source)!=before or (record_bytes is not None and (record_path.read_bytes()!=record_bytes or record['sha256']!=digest)):
            report['skipped'].append(dict(source=str(source),reason='active_atomic_checkpoint_replacement'));continue
        if digest in index['assets']:continue
        # Never accumulate hidden GBs. Count only this loop's explicitly recorded staging, not arbitrary trees.
        staged_paths={Path(v['staged']) for v in registry.values()}
        retained=sum(p.stat().st_size for p in staged_paths if p.is_file())
        if retained+before[2]>config.get('max_staged_bytes',12*2**30) or shutil.disk_usage(out).free<15*2**30:
            report['skipped'].append(dict(source=str(source),reason='staging_or_15GiB_disk_reserve'));continue
        name='%s-%d-%s-%s.pt'%(item['direction'],item['fork_seed'],('learning'+str(item['stage'])) if item['kind']=='model' else ('fullresume-learning'+str(item['stage']) if item['stage'] is not None else 'fullresume'),digest[:16])
        staged=staging/name
        if staged.exists():
            if sha(staged)!=digest:raise ValueError('Existing immutable asset conflict')
        else:
            registry[digest]=dict(staged=str(staged),source=str(source),bytes=before[2]);atomic(registry_path,registry)
            os.link(source,staged)
        # Atomic writer contract: trainer writes temp and replaces path, never mutates linked inode.
        if fingerprint_stat(source)!=before or sha(staged)!=digest or (record_bytes is not None and record_path.read_bytes()!=record_bytes):
            remove_redundant_link(staged,source)
            report['skipped'].append(dict(source=str(source),staged=str(staged),reason='replacement_race_retained_if_unique'));continue
        state=loader(staged)
        metadata=validate_metadata(state,item,param_sha,parent_sha,config['execution_mode'])
        if metadata['parent_sha256']!=parent_record['selected']['model_sha256']:raise ValueError('Parent checkpoint lineage mismatch')
        if item['kind']=='resume' and state['counters']!=record['counters']:raise ValueError('Resume sidecar counters mismatch')
        del state
        if sha(staged)!=digest:raise ValueError('Staged inode mutated; writer violates atomic contract')
        row=dict(source=name,asset_name=name,kind=item['kind'],bytes=before[2],sha256=digest,training_identity=metadata)
        plan=dict(repository=REPO,branch=BRANCH,tag=TAG,base_commit=config['base_commit'],staging_root=str(staging),files=[row])
        plan_path=out/('plan-'+digest+'.json');atomic(plan_path,plan)
        pending=dict(plan_path=str(plan_path),staged=str(staged),source=str(source),sha256=digest,bytes=before[2],training_identity=metadata)
        atomic(pending_path,pending);finish(pending);completed+=1
    atomic(out/('record-%d.sidecar.json'%time.time_ns()),report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--once',action='store_true');a=p.parse_args();config=read(a.config)
    import fcntl
    out=Path(config['model_backup_root']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'loop.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();number=0
    while time.time()<config['until_epoch']:
        try:tick(config)
        except Exception as e:atomic(out/('failure-%d.sidecar.json'%time.time_ns()),dict(epoch=time.time(),error=str(e),policy='Same pending plan reconciles remote state first; no blind overwrite or source cleanup'))
        if a.once:return
        number+=1;time.sleep(min(max(0,start+1800*number-time.monotonic()),max(0,config['until_epoch']-time.time())))
if __name__=='__main__':main()
