"""Four-case L16/L768 A development probe. No training, retries, or promotion."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from r4_protocol import analyze_rows

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser()
 for name in ('python','study-root','data-root','checkpoint','output','startup-lock'):p.add_argument('--'+name,required=True)
 p.add_argument('--node',type=int,choices=(16,768),required=True)
 p.add_argument('--timeout-seconds',type=int,default=1800)
 p.add_argument('--device',default='cuda:0')
 a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
 model=Path(a.checkpoint).resolve();fingerprint=sha(model);rows=[];deadline=time.monotonic()+a.timeout_seconds
 for seed in (261,262):
  for execution in ('D','S'):
   out=root/f'{seed}-{execution}';out.mkdir()
   cmd=[a.python,str(Path(__file__).with_name('evaluate_sac_r9c_a.py')),'--study-root',a.study_root,'--root',a.data_root,'--checkpoint',str(model),'--output_dir',str(out),'--eval_seed',str(seed),'--seed-split','development','--condition','standard','--execution',execution,'--factor','F01','--policy-noise','on','--zero-reset-wrench','--disable-interval-push','--headless','--device',a.device,'--startup-lock',a.startup_lock]
   if time.monotonic()>=deadline:raise TimeoutError('Probe budget expired')
   with (out/'process.log').open('w') as log:
    r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=deadline-time.monotonic())
   if r.returncode or not (out/'status.json').exists() or read(out/'status.json')['status']!='passed':raise RuntimeError('Technical failure: '+str(out))
   assert sha(model)==fingerprint
   assert read(out/'frozen-before.json')==read(out/'frozen-after.json')
   cfg=read(out/'configuration.json');assert cfg['sha256']==fingerprint and cfg['evaluation_seed']==seed and cfg['execution']==execution
   data=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines() if x]
   for x in data:
    assert len(x['raw_policy_history_before'])==10 and all(len(y)==33 for y in x['raw_policy_history_before'])
    assert len(x['actor_features_before'])==52 and x['critic_forward_performed'] is False
   result=analyze_rows(data)
   assert result==read(out/'protocol-result.json')
   rows.append(dict(seed=seed,execution=execution,path=str(out),result=result))
   (root/'probe-results.json').write_text(json.dumps(dict(status='running',node=a.node,model_sha256=fingerprint,cases=rows),indent=2,allow_nan=False))
 (root/'probe-results.json').write_text(json.dumps(dict(status='completed',node=a.node,development_probe_only=True,qualification_for_B=False,model_sha256=fingerprint,cases=rows),indent=2,allow_nan=False))
 print(json.dumps({'completed_cases':4,'control_passes':sum(x['result']['task_pass'] for x in rows),'index':str(root/'probe-results.json')}))
if __name__=='__main__':main()
