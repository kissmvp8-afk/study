"""单GPU顺序执行128环境候选；worker由Dot接到现有Isaac入口，CPU控制器不导入Isaac。"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from selection import assess_manifest

PLAN = 'SAC-R4-128-SCREEN-20261108'


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def timestamp(text):
    value = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None: raise ValueError('UTC offset required')
    return value.timestamp()


def _main(runtime_path):
    runtime = json.loads(Path(runtime_path).read_text())
    root = Path(runtime['workroot']).resolve(); root.mkdir(parents=True, exist_ok=True)
    budget = json.loads((root/runtime['budget_path']).read_text())
    if budget['status'] != 'ready': raise ValueError('Budget not ready')
    started = timestamp(runtime['batch_started_at_utc']); lease = timestamp(runtime['lease_expires_at_utc'])
    hard = min(started+budget['hard_cap_s'], started+34200, lease-5400)
    archive_deadline = hard-1200
    # Keep time for final primary/fallback standard/turn/pulse evaluation.
    final_reserve = max(1800., 48*runtime['eval_case_seconds']*1.15)
    training_deadline = archive_deadline-final_reserve
    command = runtime['worker_command']
    if not isinstance(command, list) or not command or not all(isinstance(x,str) for x in command):
        raise ValueError('worker_command must be an explicit argv array')
    if runtime['gpu_preflight_passed'] is not True: raise ValueError('Server preflight incomplete')
    state_path = root/'controller-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(plan_id=PLAN,status='running',completed_round=0)
    if state['plan_id'] != PLAN: raise ValueError('Workroot belongs to another task')
    if state['status'] in ('completed','budget_stopped','failed','regression_stopped'):
        print('Terminal task already recorded; use a new authorized run_id to rerun.'); return
    runtime_sha = hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest()
    if 'runtime_sha256' in state and state['runtime_sha256'] != runtime_sha:
        raise ValueError('Runtime changed on recovery; preserve the frozen batch contract')
    state['runtime_sha256'] = runtime_sha
    atomic_json(state_path,state)
    requests = root/'requests'; requests.mkdir(exist_ok=True)

    def job(name, request, deadline):
        # Leave polling and SIGTERM-save grace inside the externally visible deadline.
        worker_deadline = deadline-75
        request_path = requests/(name+'.request.json'); response_path = requests/(name+'.response.json')
        request = dict(request,plan_id=PLAN,workroot=str(root),deadline_utc=dt.datetime.fromtimestamp(worker_deadline,dt.timezone.utc).isoformat())
        # Completed responses are reused; worker must verify request/model hashes on recovery.
        if request_path.exists() and json.loads(request_path.read_text()) != request:
            raise ValueError('Immutable job request changed: '+name)
        if not request_path.exists(): atomic_json(request_path,request)
        if response_path.exists(): return json.loads(response_path.read_text())
        if time.time() >= worker_deadline: raise TimeoutError('No time left for '+name)
        state.update(active_job=name,heartbeat_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(state_path,state)
        with (requests/(name+'.process.log')).open('a') as log:
            proc = subprocess.Popen(command+['--request',str(request_path),'--response',str(response_path)],
                                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                state['heartbeat_utc'] = dt.datetime.now(dt.timezone.utc).isoformat(); atomic_json(state_path,state)
                disk_floor = (1 if request['action']=='archive' else 10)*2**30
                if time.time() >= worker_deadline or shutil.disk_usage(root).free < disk_floor:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                    raise TimeoutError('Deadline/disk guard stopped '+name)
                time.sleep(10)
            if proc.returncode != 0 or not response_path.exists(): raise RuntimeError('Worker failed: '+name)
        return json.loads(response_path.read_text())

    directions = ['D'+str(i) for i in range(6)]; q = budget['q']; decisions = []
    last_decision = None; stop_status = 'completed'
    try:
        for round_number, learning in enumerate([q,5*q//2,4*q,6*q],1):
            arms = []; jobs = [(d,s) for d in directions for s in (1001,1002)]
            # Fixed rotation changes GPU order across rounds without comparing unequal budgets.
            offset = (round_number-1) % len(jobs); jobs = jobs[offset:]+jobs[:offset]
            for d, seed in jobs:
                existing_response = requests/f'r{round_number}-{d}-{seed}.response.json'
                if not existing_response.exists():
                    if time.time() >= min(training_deadline,lease-7200) or shutil.disk_usage(root).free < 15*2**30:
                        raise TimeoutError('Insufficient time/disk to start another training job')
                response = job(f'r{round_number}-{d}-{seed}',dict(action='train_evaluate',direction=d,fork_seed=seed,
                               round=round_number,q=q,target_learning_macros=learning,
                               resume_latest_128env=round_number>1,parameters_path=runtime['parameters_path']),training_deadline)
                arms.append(response['arm'])
            manifest = dict(plan_id=PLAN,round=round_number,q=q,expected_directions=directions,arms=arms)
            manifest_path = root/f'round-{round_number}.manifest.json'; atomic_json(manifest_path,manifest)
            last_decision = assess_manifest(manifest_path)
            decision_path = root/f'round-{round_number}.decision.json'; atomic_json(decision_path,last_decision)
            decisions.append(str(decision_path.relative_to(root)))
            state.update(completed_round=round_number,last_decision=decisions[-1]); atomic_json(state_path,state)
            # Worker returns reference-comparison diagnostics; controller only uses recomputed decisions.
            if round_number == 2:
                best = last_decision['ranking'][0]
                reference = json.loads((root/runtime['parent_reference_path']).read_text())
                if best['overall_pass'] == 0 and all(r['overall_motion'] < 1/3 for r in last_decision['ranking']) and best['sort_key'][:7] >= reference['sort_key'][:7]:
                    stop_status = 'regression_stopped'; break
            directions = last_decision['promoted']
        if state['completed_round'] == 4 and stop_status == 'completed':
            champion = last_decision['promoted'][0]; best_key = last_decision['ranking'][0]['sort_key']; stagnant = 0
            for extra in range(1,5):
                # Extend both forks together only when time remains after the complete core comparison.
                forecast = q*budget['conservative_macro_s'] + 12*runtime['eval_case_seconds']*1.15 + 2*runtime.get('startup_seconds',20)
                if time.time()+forecast > min(training_deadline,started+32400-final_reserve-1200) or time.time() >= lease-7200:
                    break
                if shutil.disk_usage(root).free < 15*2**30: break
                arms = []
                for seed in (1001,1002):
                    response = job(f'extra-{extra}-{champion}-{seed}',dict(action='train_evaluate',direction=champion,
                                   fork_seed=seed,round=4,q=q,extra_champion_chunks=extra,
                                   target_learning_macros=6*q+extra*q//2,resume_latest_128env=True,
                                   parameters_path=runtime['parameters_path']),training_deadline)
                    arms.append(response['arm'])
                path = root/f'extra-{extra}.manifest.json'
                atomic_json(path,dict(plan_id=PLAN,round=4,q=q,extra_champion_chunks=extra,
                                     expected_directions=[champion],arms=arms))
                last_decision = assess_manifest(path); decision = root/f'extra-{extra}.decision.json'; atomic_json(decision,last_decision)
                decisions.append(str(decision.relative_to(root)))
                key = last_decision['ranking'][0]['sort_key']
                if key < best_key: best_key = key; stagnant = 0
                else: stagnant += 1
                state.update(extra_chunks_completed=extra,last_decision=decisions[-1]); atomic_json(state_path,state)
                if stagnant >= 2: break
    except TimeoutError as exc:
        stop_status = 'budget_stopped'; state['stop_reason'] = str(exc)
    except Exception as exc:
        stop_status = 'failed'; state['stop_reason'] = type(exc).__name__+': '+str(exc); atomic_json(state_path,state)
    if last_decision is not None and stop_status != 'failed' and time.time() < archive_deadline-75:
        try:
            final = job('final-validation',dict(action='final_validate',decision_paths=decisions,
                        parameters_path=runtime['parameters_path'],freeze_selection_before_heldouts=True),archive_deadline)
            state['final_validation'] = final
        except TimeoutError as exc:
            stop_status='budget_stopped';state['final_validation_error']=str(exc)
        except Exception as exc:
            stop_status='failed';state['final_validation_error']=type(exc).__name__+': '+str(exc)
    # Archive worker must return verified external backup facts, never infer them from an attempted upload.
    if time.time() < hard-75:
        try:
            state['archive'] = job('archive',dict(action='archive',task_status=stop_status,
                                  parameters_path=runtime['parameters_path']),hard)
        except Exception as exc:
            state['archive_error']=type(exc).__name__+': '+str(exc)
    else:
        state['archive_error']='No archive window remains; external backup unconfirmed'
    state.update(status=stop_status,active_job=None,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(state_path,state)
    print(json.dumps({'status':stop_status,'completed_round':state['completed_round']},ensure_ascii=False))


def main(runtime_path):
    # A context manager releases the process lock on every terminal or error path.
    import fcntl
    runtime = json.loads(Path(runtime_path).read_text())
    root = Path(runtime['workroot']).resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _main(runtime_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('runtime'); args = p.parse_args(); main(args.runtime)
