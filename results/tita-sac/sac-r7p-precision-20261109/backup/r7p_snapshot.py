"""Publish explicit small R7P code/config/progress snapshots; never weights or credentials."""
import argparse,json,pathlib,time,os
from r7p_backup_ops import atomic,read,sha
from r7p_publish import publish,scan_file,REPO,BRANCH,TAG,PREFIX

def run(config):
 root=pathlib.Path(config['workroot']).resolve();out=root/'evidence-backup';out.mkdir(exist_ok=True)
 import fcntl
 lock=(out/'snapshot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 epoch=time.time_ns();stage=out/('snapshot-'+str(epoch));stage.mkdir();rows=[];total=0
 allowed={'.py','.json','.jsonl','.txt','.md','.yaml','.yml','.csv','.log'}
 candidates=[]
 for p in root.rglob('*'):
  if not p.is_file() or p.is_symlink() or p.suffix not in allowed:continue
  rel=p.relative_to(root)
  if any(x.startswith('.') or x in ('__pycache__','model-backup','evidence-backup','backups') for x in rel.parts):continue
  if p.stat().st_size>4*2**20:continue
  candidates.append((str(rel),p))
 for rel,p in sorted(candidates):
  data=p.read_bytes();total+=len(data)
  if total>22*2**20:raise ValueError('Snapshot exceeds explicit 22MiB budget')
  d=stage/rel;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(data);scan_file(d)
  rows.append(dict(source=rel,destination=PREFIX+rel,bytes=len(data),sha256=sha(d),kind='git'))
 manifest=dict(epoch=time.time(),cutoff_semantics='per-file captured bytes; append logs can differ from later local state',files=[{k:v for k,v in x.items() if k!='kind'} for x in rows])
 atomic(stage/'SNAPSHOT_MANIFEST.json',manifest)
 p=stage/'SNAPSHOT_MANIFEST.json';rows.append(dict(source=p.name,destination=PREFIX+p.name,bytes=p.stat().st_size,sha256=sha(p),kind='git'))
 plan=dict(repository=REPO,branch=BRANCH,tag=TAG,base_commit=config['base_commit'],staging_root=str(stage),files=rows)
 target=out/('plan-'+str(epoch)+'.json');atomic(target,plan)
 result=publish(target,config['gh'],out/'receipts');atomic(out/'latest.json',dict(status=result['status'],commit_url=result['commit_url'],files=len(rows),epoch=time.time()))
 print(json.dumps(read(out/'latest.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(read(a.config))
