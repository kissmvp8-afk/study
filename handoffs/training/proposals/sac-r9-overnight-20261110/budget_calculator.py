"""四路并行墙钟预算估算；数字是吞吐推算，不是已完成训练结果。"""
import argparse,json,math
p=argparse.ArgumentParser();p.add_argument('--macro-seconds',type=float,default=6.7);p.add_argument('--safety-factor',type=float,default=1.15);a=p.parse_args()
assert a.macro_seconds>0 and a.safety_factor>=1
t=a.macro_seconds*a.safety_factor
def blocks(seconds,reserve):return max(0,int((seconds-reserve)/t)//64*64)
ka=blocks(150*60,50*60);kb=blocks(270*60,80*60)
out=dict(per_trajectory_macro_seconds=a.macro_seconds,forecast_seconds=t,four_parallel_slots=4,
 phase_A_learning_macros_per_trajectory=ka,phase_B_learning_macros_per_trajectory=kb,
 phase_A_budget_minutes=150,phase_B_budget_minutes=270,
 total_campaign_transitions=4096*(ka+kb+32),total_critic_updates=512*(ka+kb),total_actor_updates=256*(ka+kb),
 old_uncompressed_diagnostic_GiB=4*(ka+kb)*2145000/2**30,
 assumed_stream_compressed_GiB_if_ratio015=4*(ka+kb)*2145000*.15/2**30,
 note='Four concurrent branches take approximately max(branch_time), not sum. PhaseB has four new forks/warmstarts; retained winning lineage differs from total campaign compute. Actual targets frozen from calibration then common64-block elastic extension before final selection; no minimum-duration sleep.')
print(json.dumps(out,ensure_ascii=False,indent=2))
