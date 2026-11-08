"""根据已同步的满池吞吐和真实租期，分配整夜逐级筛选的共同训练量。"""
import argparse
import json
import math


def allocate(macro_seconds,eval_case_seconds,prep_elapsed_s,lease_remaining_s,startup_seconds=20,collect_seconds=.36):
    """q为16的倍数；工作量31q学习宏轮，首轮12臂各预填充16宏轮。"""
    values=(macro_seconds,eval_case_seconds,prep_elapsed_s,lease_remaining_s,startup_seconds,collect_seconds)
    if not all(math.isfinite(v) and v>=0 for v in values) or macro_seconds<=0 or eval_case_seconds<=0:
        raise ValueError('Invalid measured profile or lease')
    hard=min(34200.,prep_elapsed_s+lease_remaining_s-5400.)
    target=min(32400.,hard-300)
    reserve_eval=max(4500.,180*eval_case_seconds*1.15)
    reserve_archive=1200.;reserve_recovery=600.
    launch=24*startup_seconds+12*16*collect_seconds
    fixed=prep_elapsed_s+reserve_eval+reserve_archive+reserve_recovery+launch
    candidates=[]
    conservative_macro=macro_seconds*1.08
    for q in range(64,2049,16):
        total=fixed+31*q*conservative_macro
        if total<=hard:candidates.append((abs(total-target),q,total))
    if not candidates:
        return dict(status='blocked_insufficient_time_for_fair_minimum',hard_cap_s=hard,
                    estimated_fixed_s=fixed,minimum_learning_macros_per_arm=64)
    _,q,total=min(candidates)
    levels=[q,5*q//2,4*q,6*q]
    return dict(status='ready',mode='full_overnight' if prep_elapsed_s+lease_remaining_s>=39600 else 'shortened_by_actual_lease',
                q=q,target_wall_s=target,hard_cap_s=hard,predicted_wall_s=total,
                measured_macro_s=macro_seconds,conservative_macro_s=conservative_macro,
                reserved_eval_s=reserve_eval,reserved_archive_s=reserve_archive,reserved_recovery_s=reserve_recovery,
                cumulative_learning_macros=levels,directions_remaining=[6,3,2,1],repeats_per_direction=2,
                critic_updates_per_surviving_arm=[128*x for x in levels],
                new_transitions_per_surviving_arm=[1024*(16+x) for x in levels],
                total_scheduled_learning_macros=31*q,total_scheduled_critic_updates=31*q*128,
                total_scheduled_new_transitions=(31*q+12*16)*1024,
                extra_champion_chunks_max=4,extra_chunk_learning_macros_per_repeat=q//2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--macro-seconds',type=float,required=True);p.add_argument('--eval-case-seconds',type=float,required=True)
    p.add_argument('--prep-elapsed-seconds',type=float,required=True);p.add_argument('--lease-remaining-seconds',type=float,required=True)
    p.add_argument('--startup-seconds',type=float,default=20);p.add_argument('--collect-seconds',type=float,default=.36)
    a=p.parse_args();print(json.dumps(allocate(a.macro_seconds,a.eval_case_seconds,a.prep_elapsed_seconds,
                                             a.lease_remaining_seconds,a.startup_seconds,a.collect_seconds),indent=2))
