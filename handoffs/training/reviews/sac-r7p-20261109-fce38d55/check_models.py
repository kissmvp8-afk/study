"""只读核验远端最终模型/恢复点，不启动训练或Isaac。"""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
inp=json.loads((R/'input.json').read_text());resumes=json.loads((R/'resume-assets.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def download(a):
 p=R/a['name']
 if not p.exists():
  with urllib.request.urlopen(a['url'],timeout=60) as s,p.open('wb') as f:
   while b:=s.read(1024*1024):f.write(b)
 assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256']
 return a
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 results=list(ex.map(download,resumes))
def finite(o):
 if torch.is_tensor(o):
  if o.is_floating_point():assert torch.isfinite(o).all().item()
  return 1
 if isinstance(o,dict):return sum(finite(v) for v in o.values())
 if isinstance(o,(tuple,list)):return sum(finite(v) for v in o)
 return 0
records=[]
for a in inp['models']+resumes:
 p=R/a['name'];assert sha(p)==a['sha256']
 d=torch.load(p,map_location='cpu',weights_only=True)
 tensors=finite(d)
 c=d['counters'];assert c['iterations']==64 and c['transitions']==65536
 assert c['critic_updates']==c['alpha_updates']==c['target_updates']==c['auxiliary_updates']==6144 and c['actor_updates']==3072
 assert c['auxiliary_skipped']==0
 assert d['optimizers']['actor']['param_groups'][0]['lr']==5e-5
 alpha=float(d['modules']['temperature']['log_alpha'].exp())
 record={'name':a['name'],'sha256':a['sha256'],'bytes':p.stat().st_size,'status':'downloaded_SHA_size_weights_only_CPU_finite_passed','finite_tensor_count':tensors,'alpha':alpha,'counters':c,'replay_present':d['replay'] is not None,'environment_state_saved':d['environment_state_saved'],'actual_training_continuation':'not_run','optimizer_steps':{k:sorted({float(v['step']) for v in o['state'].values() if isinstance(v,dict) and 'step' in v}) for k,o in d['optimizers'].items()}}
 if d['replay'] is not None:record['replay_keys']=list(d['replay'])
 records.append(record);print(a['name'],'verified',flush=True)
 del d
(R/'model-and-resume-verification.json').write_text(json.dumps(records,indent=2)+'\n')
print('all final4 models and4 fullresumes independently downloaded/read; actual training restore not run',flush=True)
