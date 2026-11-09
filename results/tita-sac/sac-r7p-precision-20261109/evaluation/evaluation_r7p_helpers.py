"""Pure protocol helpers shared by evaluator and CPU tests."""
import hashlib
import json

SEED_SPLITS={'development':(141,142,143)}

def validate_seed(split, seed, condition):
    assert split=='development' and seed in SEED_SPLITS[split], 'R7P only development141/142/143'
    assert condition=='standard', 'R7P does not run heldout/turn/pulse/other diagnostics'

def physical_fingerprint(initial):
    required={'root_pos_w','root_com_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w',
              'material_properties','masses','inertias','coms','explicit_actuators',
              'joint_pos','joint_vel','stiffness','damping','external_force_b','external_torque_b'}
    assert set(initial)==required, 'Physical fingerprint schema mismatch'
    return hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def current_events(env):
    # ManagerBasedRLEnv.step computes these once before calling the reward manager.
    # Do not re-run termination functions or read retained per-term get_term buffers.
    assert set(env.termination_manager.active_terms)=={'base_contact','time_out'}
    return {'base_contact':bool(env.reset_terminated[0]),'time_out':bool(env.reset_time_outs[0])}


def schedule(condition, i):
    # env.step computes one next observation after the final logged transition.
    # Permit that endpoint command update without adding a sampled/logged step.
    assert 0 <= i <= (600 if condition == 'standard' else 900)
    if condition == 'diagnostic_zero': return 'zero', (0.,0.,0.)
    if condition == 'diagnostic_switch':
        if i < 100: return 'idle_before', (0.,0.,0.)
        if i < 300: return 'forward', (.3,0.,0.)
        if i < 600: return 'idle_between', (0.,0.,0.)
        if i < 800: return 'reverse', (-.3,0.,0.)
        return 'idle_after', (0.,0.,0.)
    if condition == 'pulse': return ('pulse' if 200<=i<210 else 'zero'), (0.,0.,0.)
    if condition == 'turn':
        if i<100: return 'idle_before',(0.,0.,0.)
        if i<300: return 'turn_left',(.2,0.,.5)
        if i<400: return 'idle_between',(0.,0.,0.)
        if i<600: return 'turn_right',(.2,0.,-.5)
        return 'idle_after',(0.,0.,0.)
    assert condition == 'standard'
    if i<100:return 'idle_before',(0.,0.,0.)
    if i<300:return 'forward',(.3,0.,0.)
    if i<400:return 'idle_between',(0.,0.,0.)
    return 'reverse',(-.3,0.,0.)


def diagnostic_result(rows, condition):
    from r4_protocol import finite, close, window
    assert condition in ('diagnostic_zero','diagnostic_switch')
    assert rows and len(rows)<=900 and finite(rows)
    for i,r in enumerate(rows):
        assert r['step']==i and abs(r['t_start_s']-i*.02)<1e-8 and abs(r['t_end_s']-(i+1)*.02)<1e-8
        assert close(r['effective_command'],schedule(condition,i)[1])
        assert r['termination_terms']=={'base_contact':r['terminated'],'time_out':r['time_out']}
        assert not i<len(rows)-1 or not (r['terminated'] or r['time_out'])
    assert len(rows)==900 or rows[-1]['terminated'] or rows[-1]['time_out'], 'Incomplete diagnostic evidence'
    windows=[(0,2),(2,6),(6,12),(12,18),(7,8),(10,12)]
    complete=len(rows)==900 and not any(r['terminated'] or r['time_out'] for r in rows)
    return {'protocol':condition,'diagnostic_only':True,'task_pass':None,'complete':complete,
            'survived':complete and max(r['contact_named_max_N']['base'] for r in rows)<=1.,
            'windows':{f'{a}-{b}':window(rows,a,b) for a,b in windows},
            'standard_acceptance_unchanged':True}
