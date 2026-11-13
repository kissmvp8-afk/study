"""R1 逐步证据统计；标准库，无仿真副作用。缺失为 null，不补零。"""
import math, json, argparse, csv
from pathlib import Path

def windows(protocol):
    if protocol.startswith('P'):return [('settle',0,2),('stable_2_10',2,10),('stable_10_18',10,18),('last_1s',17,18)]
    spans=[('idle_before',0,2),('turn_left' if protocol=='turn' else 'forward',2,6),('idle_between',6,8),('turn_right' if protocol=='turn' else 'reverse',8,12)]
    if protocol=='turn':spans += [('idle_after',12,18),('last_1s',17,18)]
    return spans+[('idle_before_tail',1,2),('idle_between_tail',7,8),('stable_all_after_2',2,18 if protocol=='turn' else 12)]

def stage_metrics(records,protocol='standard'):
    result=[]
    for name,start,end in windows(protocol):
        rows=[r for r in records if r['t_start_s']>=start-1e-8 and r['t_end_s']<=end+1e-8]
        duration=sum(r['t_end_s']-r['t_start_s'] for r in rows)
        complete=bool(rows) and abs(duration-(end-start))<1e-6 and abs(rows[0]['t_start_s']-start)<1e-6 and abs(rows[-1]['t_end_s']-end)<1e-6
        v=dict(stage=name,start_s=start,end_s=end,samples=len(rows),observed_s=duration,complete=complete)
        fields=['mean_abs_vx','mean_abs_vy','mean_vx','mean_vy','mean_wz','mean_planar_speed','vx_mae','wz_mae','mean_abs_wz','net_drift_m','path_length_m','heading_change_rad','mean_height_m','minimum_height_m','mean_tilt_rad','maximum_tilt_rad','actor_action_rate_per_s','leg_target_rate_rad_s','wheel_target_rate_rad_s2']
        fields += ['contact_'+k+'_fraction' for k in ('base','left_leg_3','right_leg_3','left_wheel','right_wheel')]
        v.update({k:None for k in fields})
        if rows:
            mean=lambda f:sum(f(r) for r in rows)/len(rows)
            vel=lambda r:r['measured_body_vx_vy_wz']
            tilt=lambda r:math.acos(max(-1.,min(1.,-r['gravity_b'][2])))
            v.update(mean_abs_vx=mean(lambda r:abs(vel(r)[0])),mean_abs_vy=mean(lambda r:abs(vel(r)[1])),mean_vx=mean(lambda r:vel(r)[0]),mean_vy=mean(lambda r:vel(r)[1]),mean_wz=mean(lambda r:vel(r)[2]),mean_planar_speed=mean(lambda r:math.hypot(*vel(r)[:2])),vx_mae=mean(lambda r:abs(vel(r)[0]-r['effective_command'][0])),wz_mae=mean(lambda r:abs(vel(r)[2]-r['effective_command'][2])),mean_abs_wz=mean(lambda r:abs(vel(r)[2])),mean_height_m=mean(lambda r:r['root_pos_w'][2]),minimum_height_m=min(r['root_pos_w'][2] for r in rows),mean_tilt_rad=mean(tilt),maximum_tilt_rad=max(map(tilt,rows)),heading_change_rad=sum(math.atan2(math.sin(r['heading_w']-r['heading_w_before']),math.cos(r['heading_w']-r['heading_w_before'])) for r in rows),net_drift_m=math.dist(rows[0]['root_com_pos_w_before'][:2],rows[-1]['root_com_pos_w'][:2]),path_length_m=sum(math.dist(r['root_com_pos_w_before'][:2],r['root_com_pos_w'][:2]) for r in rows))
            for key,outkey in [('actor_action','actor_action_rate_per_s'),('leg_target_rad','leg_target_rate_rad_s'),('wheel_target_rad_s','wheel_target_rate_rad_s2')]:
                if len(rows)>1:v[outkey]=sum(sum(abs(x-y) for x,y in zip(b[key],a[key]))/len(b[key])/(b['t_end_s']-a['t_end_s']) for a,b in zip(rows,rows[1:]))/(len(rows)-1)
            for k in ('base','left_leg_3','right_leg_3','left_wheel','right_wheel'):
                if all(k in r.get('contact_named_max_N',{}) for r in rows):v['contact_'+k+'_fraction']=mean(lambda r:float(r['contact_named_max_N'][k]>1))
        result.append(v)
    return result

def d_gate(records,metrics):
    m={r['stage']:r for r in metrics}
    return bool(len(records)==600 and not any(r['terminated'] or r['time_out'] for r in records) and all(m[k]['complete'] for k in ('idle_before','forward','idle_between','reverse')) and all(m[k]['vx_mae']<=.12 and m[k]['mean_abs_wz']<=.20 for k in ('forward','reverse')) and m['idle_between_tail']['mean_planar_speed']<=.05 and m['stable_all_after_2']['mean_height_m']>=.25 and all(r.get('contact_named_max_N',{}).get('base',float('inf'))<=1 for r in records))

def paired_diagnostics(directories):
    """同 seed 四条件做显式初态核对；差值仅在匹配时有效，不静默补缺。"""
    grouped={}
    for directory in directories:
        directory=Path(directory)
        cfg=json.loads((directory/'configuration.json').read_text())
        summary=json.loads((directory/'summary.json').read_text())
        grouped.setdefault(str(cfg['evaluation_seed']),{})[cfg['condition']]=(cfg,summary)
    result=[]
    physical=('root_pos_w','root_com_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel','material_properties','masses','inertias','coms','explicit_actuators')
    for seed,arms in grouped.items():
        matched={};diffs=[]
        baseline=arms.get('P00')
        for condition in ('P10','P01','P11'):
            other=arms.get(condition)
            fields={key:bool(baseline and other and key in baseline[0]['initial_state'] and key in other[0]['initial_state'] and baseline[0]['initial_state'][key]==other[0]['initial_state'][key]) for key in physical}
            fields['reset_wrench_before_intervention']=bool(baseline and other and baseline[0]['reset_wrench_before_intervention']==other[0]['reset_wrench_before_intervention'])
            matched[condition]=fields
        for window in ('settle','stable_2_10','stable_10_18','last_1s'):
            values={k:next((m for m in v[1].get('stage_metrics',[]) if m['stage']==window),None) for k,v in arms.items()}
            valid=len(values)==4 and all(all(v.values()) for v in matched.values()) and all(v and v['complete'] for v in values.values()) and all(v[1]['status']=='passed' for v in arms.values())
            for metric in ('mean_planar_speed','mean_abs_vx','mean_abs_vy','net_drift_m','path_length_m','actor_action_rate_per_s'):
                row=dict(seed=seed,window=window,metric=metric,paired_complete=valid,noise_off_delta=None,wrench_off_delta=None,both_off_delta=None,interaction=None)
                if valid and all(v[metric] is not None for v in values.values()):
                    a,b,c,d=[values[k][metric] for k in ('P00','P10','P01','P11')]
                    row.update(noise_off_delta=b-a,wrench_off_delta=c-a,both_off_delta=d-a,interaction=d-b-c+a)
                diffs.append(row)
        result.append(dict(seed=seed,physical_match_fields=matched,contrasts=diffs))
    trigger_seeds=[]
    for seed,arms in grouped.items():
        if 'P00' not in arms:continue
        summary=arms['P00'][1];m={r['stage']:r for r in summary.get('stage_metrics',[])}
        if summary['status']=='passed' and all(k in m and m[k]['complete'] and m[k]['mean_planar_speed']>.05 for k in ('stable_2_10','stable_10_18')):trigger_seeds.append(seed)
    return dict(paired_diagnostics=result,s1_trigger_seeds=trigger_seeds,s1_trigger=len(trigger_seeds)>=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('steps',type=Path,nargs='*');p.add_argument('--diagnostic-roots',type=Path,nargs='+');p.add_argument('--protocol',default='standard');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.diagnostic_roots:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(paired_diagnostics(a.diagnostic_roots),indent=2,allow_nan=False)+'\n');raise SystemExit(0)
    if not a.steps:p.error('provide steps files or --diagnostic-roots')
    allrows=[]
    for path in a.steps:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        cfgpath=path.parent/'configuration.json'
        if cfgpath.exists():
            cfg=json.loads(cfgpath.read_text());names=cfg.get('contact_body_order',[])
            # 旧文件没有逐侧标签时只按已保存的精确名字映射，不把 any 接触当腿接触。
            # 原 evaluate_np3o_fixed.py 的明确输出顺序：base + E03 八个动作关节的 URDF child。
            expected={k:names[i] for k,i in {'base':0,'left_leg_3':3,'right_leg_3':7,'left_wheel':4,'right_wheel':8}.items()} if len(names)==9 and names[0]=='base_link' else {}
            for r in rows:
                if 'contact_named_max_N' not in r:r['contact_named_max_N']={k:r['contact_force_history_max_N'][names.index(v)] for k,v in expected.items() if v in names}
        allrows += [dict(source=str(path),**m) for m in stage_metrics(rows,a.protocol)]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(allrows,indent=2,allow_nan=False)+'\n')
    with a.output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
