"""Sequential subprocess evaluation and frozen, gated heldout validation for R4."""
import hashlib,json,math,os,subprocess,time
from pathlib import Path
from r4_protocol import analyze_rows
from evaluation_r7p_helpers import SEED_SPLITS

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def local(root,path):
    p=(Path(root)/path).resolve();p.relative_to(Path(root).resolve());return p

def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def make_evaluate_case(root, python, evaluator, study_root, data_root, deadline, device='cuda:0', namespace='evaluations', startup_lock=None, bootstrap_env=None):
    """Callback signature: (model_path, model_sha256, split, protocol, execution, seed, pulse_direction=None)."""
    root=Path(root).resolve();evaluator=Path(evaluator).resolve()
    namespace_root=local(root,namespace)
    bootstrap_env=dict(bootstrap_env or {})
    allowed={'R6_ORIGINAL_EVALUATOR','R6_ORIGINAL_EVALUATOR_SHA256','R6_KIT_PROVIDER_MANIFEST','R6_KIT_PROVIDER_MANIFEST_SHA256'}
    assert not bootstrap_env or set(bootstrap_env)==allowed
    if bootstrap_env:
        assert sha(bootstrap_env['R6_ORIGINAL_EVALUATOR'])==bootstrap_env['R6_ORIGINAL_EVALUATOR_SHA256']
        assert sha(bootstrap_env['R6_KIT_PROVIDER_MANIFEST'])==bootstrap_env['R6_KIT_PROVIDER_MANIFEST_SHA256']
    def evaluate_case(model_path,model_sha256,split,protocol,execution,seed,pulse_direction=None):
        model=local(root,model_path);assert sha(model)==model_sha256
        from evaluation_r7p_helpers import validate_seed
        validate_seed(split,seed,protocol)
        assert execution in ('D','S') and pulse_direction is None
        case_root=namespace_root/model_sha256/split/(protocol+'-'+execution+'-'+str(seed)+(('-'+pulse_direction) if pulse_direction else ''))
        case_root.mkdir(parents=True,exist_ok=True)
        # At most one technical retry; physical falls return successful process results.
        for attempt in (1,2):
            out=case_root/f'attempt-{attempt}'
            if out.exists():
                if (out/'status.json').exists() and read(out/'status.json')['status']=='passed':
                    if not (out/'process-result.json').exists():
                        raise RuntimeError('Uncertain evaluator process exit; inspect before cache reuse: '+str(out))
                    if read(out/'process-result.json').get('returncode')!=0:
                        raise RuntimeError('Passed core evaluation but unsuccessful process exit; inspect before cache reuse: '+str(out))
                    break
                if (out/'error.json').exists():
                    err=read(out/'error.json')
                    if err['type'] in ('AssertionError','FloatingPointError','OutOfMemoryError','ValueError') or 'out of memory' in err.get('message','').lower():
                        raise RuntimeError('Previously fatal evaluation contract failure; no retry: '+str(out))
                if not (out/'process-result.json').exists():raise RuntimeError('Uncertain evaluation attempt; inspect before resuming: '+str(out))
                continue
            if time.time()>=deadline:raise TimeoutError('Evaluation deadline reached')
            out.mkdir()
            command=[str(python),str(evaluator),'--root',str(data_root),'--study-root',str(study_root),'--checkpoint',str(model),'--output_dir',str(out),'--eval_seed',str(seed),'--seed-split',split,'--condition',protocol,'--execution',execution,'--factor','F01','--policy-noise','on','--zero-reset-wrench','--disable-interval-push','--headless','--device',device]
            if startup_lock:command+=['--startup-lock',str(startup_lock)]
            if pulse_direction:command+=['--pulse-direction',pulse_direction]
            with (out/'process.log').open('w') as log:
                try:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,deadline-time.time()),env=dict(os.environ,**bootstrap_env))
                except subprocess.TimeoutExpired:
                    atomic(out/'process-result.json',{'timed_out':True});raise TimeoutError('Evaluation deadline exceeded')
            atomic(out/'process-result.json',{'returncode':result.returncode})
            if (out/'error.json').exists():
                err=read(out/'error.json')
                if err['type'] in ('AssertionError','FloatingPointError','OutOfMemoryError','ValueError') or 'out of memory' in err.get('message','').lower():
                    raise RuntimeError('Fatal evaluation contract failure; no retry: '+str(out))
            if result.returncode==0 and (out/'status.json').exists() and read(out/'status.json')['status']=='passed':break
        else:raise RuntimeError('Both technical evaluation attempts failed: '+str(case_root))
        cfg=read(out/'configuration.json')
        from evaluation_r7p_helpers import physical_fingerprint
        assert physical_fingerprint(cfg['initial_state'])==cfg['physical_initial_state_sha256']
        for key,value in dict(sha256=model_sha256,execution=execution,evaluation_seed=seed,condition=protocol,seed_split=split,policy_noise='on',zero_reset_wrench=True,interval_push_disabled=True,tf32_matmul=False,tf32_cudnn=False,termination_event_source='current_compute_return').items():assert cfg[key]==value,(key,cfg.get(key))
        assert sha(model)==model_sha256
        assert read(out/'frozen-before.json')==read(out/'frozen-after.json')
        rows=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines() if line.strip()]
        from evaluation_r7p_helpers import diagnostic_result
        result=diagnostic_result(rows,protocol) if protocol.startswith('diagnostic_') else analyze_rows(rows,protocol,pulse_direction)
        case=dict(execution=execution,eval_seed=seed)
        for key,name in [('steps_path','steps.jsonl'),('configuration_path','configuration.json'),('frozen_before_path','frozen-before.json'),('frozen_after_path','frozen-after.json')]:case[key]=str((out/name).relative_to(root))
        return case | dict(protocol=protocol,pulse_direction=pulse_direction,result=result,model_sha256=model_sha256,physical_initial_state_sha256=cfg['physical_initial_state_sha256'])
    return evaluate_case
