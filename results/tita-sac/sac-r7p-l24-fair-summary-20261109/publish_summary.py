"""Explicit L24/L48 summary publication only; no C2 access, model upload, or automatic retry."""
import argparse,hashlib,importlib.util,json,os,sys,time,fcntl
from pathlib import Path
PREFIX='results/tita-sac/sac-r7p-l24-fair-summary-20261109/'
OLD_PREFIXES=('results/tita-sac/sac-r7p-l24-explore-20261109/','results/tita-sac/sac-r7p-l24-232233d-explore-20261109/','results/tita-sac/sac-r7p-l24-234235d-explore-20261109/')
SCOPES={'231D','231S','232233D','232233S','234235D','234235S'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));os.replace(tmp,p)
def captured(source,target):
 source=Path(source)
 if source.is_symlink() or not source.is_file():raise ValueError('Regular explicit source required')
 before=source.stat();data=source.read_bytes();after=source.stat()
 if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Source changed while capturing')
 target.write_bytes(data)
 if sha(source)!=sha(target):raise ValueError('Source capture SHA mismatch')
 return dict(original_path=str(source),bytes=len(data),sha256=sha(target))
def run(root,baseconfig,publisher_path,local_only=False):
 root=Path(root).resolve();base=root/'summary-publication';base.mkdir(exist_ok=True)
 with (base/'publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  latest=base/'latest.json'
  if latest.exists():raise ValueError('Publication already prepared/attempted; reconcile existing receipt, never blind retry')
  if (root/'PUBLICATION_HOLD.json').exists():raise ValueError('Publication hold active')
  comparison=read(root/'comparison.json');config=read(root/'config.json')
  if comparison.get('status')!='complete_fair_comparison' or comparison.get('complete') is not True:raise ValueError('Require complete fair comparison')
  if set(config['l24_roots'])!=SCOPES:raise ValueError('Require exact six L24 nodes')
  stage=base/('staging-'+str(time.time_ns()));stage.mkdir();manifest=[]
  for name in ('comparison.json','config.json','summarize_l24_l48.py'):
   manifest.append(dict(name=name,**captured(root/name,stage/name)))
  manifest.append(dict(name='publish_summary.py',**captured(Path(__file__),stage/'publish_summary.py')))
  # Persist an isolated copy of the already-reviewed publisher; only its output prefix changes.
  source=Path(publisher_path).read_text();lines=source.splitlines();matches=[i for i,x in enumerate(lines) if x.startswith('PREFIX=')]
  if len(matches)!=1:raise ValueError('Publisher must contain exactly one explicit PREFIX assignment')
  original_prefix=lines[matches[0]].split('=',1)[1].strip().strip("'\"")
  if not original_prefix.startswith('results/tita-sac/sac-r7p-l24'):raise ValueError('Only existing L24 publisher is permitted')
  lines[matches[0]]='PREFIX='+repr(PREFIX);adapted='\n'.join(lines)+'\n';module_path=base/'summary_publisher.py';module_path.write_text(adapted)
  captured(module_path,stage/'summary_publisher.py');manifest.append(dict(name='summary_publisher.py',sha256=sha(stage/'summary_publisher.py'),bytes=(stage/'summary_publisher.py').stat().st_size,original_publisher_sha256=sha(publisher_path),change='Only explicit result PREFIX assignment'))
  ops=Path(publisher_path).with_name('r7p_backup_ops.py');captured(ops,base/'r7p_backup_ops.py');captured(ops,stage/'r7p_backup_ops.py');manifest.append(dict(name='r7p_backup_ops.py',sha256=sha(stage/'r7p_backup_ops.py'),bytes=(stage/'r7p_backup_ops.py').stat().st_size))
  nodes=[]
  for scope in sorted(SCOPES):
   node=Path(config['l24_roots'][scope]).resolve()
   if 'sac-r7p-l24' not in node.name or 'c2' in str(node).lower():raise ValueError('Only explicit L24 node receipts, never C2')
   receipt=node/'archive-backup/latest.json';record=read(receipt)
   if record.get('status')!='remote_verified':raise ValueError('Node archive not remote_verified: '+scope)
   remote=record.get('receipt',{});url=remote.get('commit_url') or record.get('commit_url')
   import re
   if not isinstance(url,str) or not re.fullmatch(r'https://github.com/kissmvp8-afk/study/commit/[0-9a-f]{40}',url):raise ValueError('Missing fixed verified node commit: '+scope)
   name='node-'+scope+'-remote-receipt.json';capture=captured(receipt,stage/name);manifest.append(dict(name=name,**capture));nodes.append(dict(scope=scope,commit_url=url,commit=url.rsplit('/',1)[-1],receipt_file=name,receipt_sha256=capture['sha256']))
  atomic(stage/'node-commit-index.json',dict(nodes=nodes,scope='Six L24 nodes; no C2 access or publication',created_epoch=time.time()))
  (stage/'README.txt').write_text('L24 versus L48: 40 matched evaluation pairs\nFour fixed candidates, five seeds, D and native-std S. D/S are not independent training seeds. The comparison checks matching source, initial physics, RNG and evaluation protocol, with frozen-state and model SHA evidence.\nOriginal C1 L48 scores are read without rescoring or replacement. Incomplete trajectories keep whole-arm continuous averages NULL; survivor-only averages are not used.\nThis is controlled simulation evidence, not proof of stable real-robot operation, a causal conclusion, or permission to select/continue a model. No model weights are uploaded here.\nEach of the six source nodes is linked by a fixed verified commit and archived receipt.\n')
  for name in ('node-commit-index.json','README.txt'):manifest.append(dict(name=name,sha256=sha(stage/name),bytes=(stage/name).stat().st_size))
  atomic(stage/'SUMMARY_MANIFEST.json',dict(files=manifest,policy='Explicit summary/source/readme/six-node receipts only; no recursion, no C2, no weights',created_epoch=time.time()))
  sys.path.insert(0,str(base));spec=importlib.util.spec_from_file_location('summary_publisher',module_path);publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
  rows=[]
  for name in [x['name'] for x in manifest]+['SUMMARY_MANIFEST.json']:
   p=stage/name;publisher.scan_file(p);rows.append(dict(source=name,destination=PREFIX+name,kind='git',bytes=p.stat().st_size,sha256=sha(p)))
  cfg=read(baseconfig);plan=dict(repository=publisher.REPO,branch=publisher.BRANCH,tag=publisher.TAG,base_commit=cfg['base_commit'],staging_root=str(stage),files=rows)
  planpath=base/'publish-plan.json';atomic(planpath,plan);publisher.validate_plan(plan)
  state=dict(status='local_verified',staging=str(stage),plan_sha256=sha(planpath),manifest_sha256=sha(stage/'SUMMARY_MANIFEST.json'));atomic(latest,state)
  if local_only:return state
  state['status']='publication_pending';atomic(latest,state)
  try:
   result=publisher.publish(planpath,cfg['gh'],base/'receipts');state.update(status=result['status'],receipt=result)
   # Independent Git blob reread against every exact staged byte digest.
   (base/'readback').mkdir(exist_ok=True);api=publisher.GitHub(cfg['gh'],base/'readback');commit=result['commit_url'].rsplit('/',1)[-1];verified=[]
   import base64
   for row in rows:
    item=api.api('repos/'+publisher.REPO+'/contents/'+row['destination']+'?ref='+commit)
    if item.get('encoding')!='base64':raise ValueError('Git contents readback unavailable')
    payload=base64.b64decode(item['content']);digest=hashlib.sha256(payload).hexdigest()
    if digest!=row['sha256']:raise ValueError('Remote Git byte readback mismatch')
    verified.append(dict(path=row['destination'],sha256=digest,bytes=len(payload)))
   state['git_readback']=verified;state['status']='remote_verified_and_readback';atomic(latest,state)
  except BaseException:
   state['status']='publication_uncertain';atomic(latest,state);raise
  return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base-config',required=True);p.add_argument('--publisher',required=True);p.add_argument('--local-only',action='store_true');a=p.parse_args();print(json.dumps(run(a.root,a.base_config,a.publisher,a.local_only)))
