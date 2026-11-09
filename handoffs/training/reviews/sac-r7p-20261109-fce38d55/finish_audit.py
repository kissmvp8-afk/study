import csv,hashlib,json,math,statistics
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence';A=R/'audit'
def read(p):return json.loads(p.read_text())
def save(n,d):(A/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
g=read(A/'groups.json');reported=read(E/'endpoint-development.json')['decision']
pair={}
for d in ['P0','P1']:
 gg=[g[d+'-'+str(f)] for f in (1401,1402)]
 pair[d]={'worst_fork_pass':min(x['passed'] for x in gg),'overall_pass':sum(x['passed'] for x in gg),'worst_fork_survival':min(x['survived'] for x in gg),'worst_fork_motion':min(x['motion'] for x in gg),'P':max(x['P'] for x in gg),'yaw':max(x['yaw'] for x in gg),'moving_mae':max(x['moving_mae'] for x in gg),'pooled_P':statistics.mean(x['P'] for x in gg)}
 a=pair[d];b=reported[d]
 for k,j in [('worst_fork_pass','worst_fork_pass_count'),('overall_pass','overall_pass_count'),('P','parking_P'),('yaw','parking_yaw'),('moving_mae','moving_mae')]:assert abs(a[k]-b[j])<1e-12
a,b=pair['P0'],pair['P1']
required=[b['worst_fork_survival']==6,b['worst_fork_motion']==6,b['worst_fork_pass']>=a['worst_fork_pass'],b['overall_pass']>=a['overall_pass'],b['moving_mae']<=.12,b['moving_mae']<=a['moving_mae']+.01,b['yaw']<=max(.10,a['yaw'])]
benefit=[b['worst_fork_pass']>=a['worst_fork_pass']+1,b['P']<=.8*a['P'] and b['P']<=g['parent-1201']['P']]
promising=all(required) and any(benefit)
assert promising==reported['promising_short_budget']
save('decision-recomputed.json',{'pair':pair,'promising_short_budget':promising,'checks':required,'benefit':benefit,'inconsistent_parking_fork_signs':True,'worst_fork_P_relative_change':b['P']/a['P']-1,'pooled_P_relative_change':b['pooled_P']/a['pooled_P']-1,'P1_pooled_task_pass_fraction':b['overall_pass']/12,'independent_acceptance':False,'direction_reproduced_on_unseen_seeds':'not_tested'})
gpu=read(E/'resources/learning-gpu-summary.json');samples=[]
with (E/'resources/gpu-samples.csv').open() as f:
 for row in csv.DictReader(f):
  row={k.strip():v.strip() for k,v in row.items()}
  ts=datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone(timedelta(hours=8))).timestamp()
  if gpu['start_epoch']<=ts<=gpu['end_epoch']:samples.append(float(row['utilization.gpu [%]'].replace('%','')))
assert len(samples)==61 and abs(statistics.mean(samples)-gpu['mean'])<1e-10 and min(samples)==52 and sum(x<80 for x in samples)==1
c=read(E/'controller-state.json');w=read(E/'endpoint-watch-state.json')
elapsed_eval=w['evaluation_result']['ended_epoch']-w['evaluation_launch_epoch']
assert elapsed_eval>0
percase=[x['summary_timing']['elapsed_s'] for x in read(A/'evaluation-time-fields.json')]
timing={'training_all_worker_start_to_learning_complete_s':c['elapsed_seconds'],'training_started_UTC':datetime.fromtimestamp(c['started_epoch'],timezone.utc).isoformat(),'training_completed_UTC':datetime.fromtimestamp(c['completed_epoch'],timezone.utc).isoformat(),'all30_evaluation_launch_to_complete_s':elapsed_eval,'all30_evaluation_s_per_case_amortized':elapsed_eval/30,'rollout_summary_elapsed_p75_s_excludes_process_initialization':statistics.quantiles(percase,n=4)[2],'GPU_common_learning_window_s':gpu['end_epoch']-gpu['start_epoch'],'GPU_samples':len(samples),'GPU_mean':statistics.mean(samples),'GPU_min':min(samples),'GPU_below80':sum(x<80 for x in samples),'finish_train_plus_endpoint_s':w['evaluation_result']['ended_epoch']-c['started_epoch'],'archive_fullZIP_build_s':read(R/'input.json')['local_verification']['created_epoch']-read(R/'input.json')['raw_manifest']['created_epoch'],'note':'423s is only startup plus training; not total evidence/backup/readback queue. Earlier135min short-trial forecast is not a measured bound for this run.'}
save('performance.json',timing)
inputs=read(R/'input.json');manifest=inputs['raw_manifest'];save('archive-verification.json',{'bytes':194219707,'sha256':'9ae8adc772aead8a4ecd510d8566d300a27089836b5d94af02b608383ea1fab3','CRC':'passed','members_SHA_size_verified':manifest.get('file_count',len(manifest.get('files',[]))),'pieces':9,'postprefill_pairing':'two files separately fetched from fixed Git snapshot; omitted from raw archive inclusion but preserved in Git','partial_git_controller_status':'running','full_raw_controller_status':c['status'],'report_and_raw_fixed_commits':[inputs['input_commit'],'986c9781afd0c0ed56ffb4e22e5e0c382e6d831e']})
pairing=read(R/'postprefill-pairing-from-fixed-git.json')
for item in pairing:
 e=read(next((E/'arms'/('P0-'+str(item['seed']))).glob('entry-fingerprints-*.json')))
 for key in ['modules','optimizers','normalizers']:assert e[key]==item['content'][key]
save('postprefill-pairing.json',pairing)
save('model-and-resume-verification.json',read(R/'model-and-resume-verification.json'))
print(json.dumps({'decision':promising,'pairs':pair,'performance':timing},ensure_ascii=False,indent=2))
