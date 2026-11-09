"""Single serialized backup attempt; model assets first, then code/log snapshot."""
import argparse,fcntl,json,pathlib,time
from r7p_backup_ops import read,atomic
from r7p_model_backup_loop import tick
from r7p_snapshot import run
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=read(a.config);r=pathlib.Path(c['workroot']);lock=(r/'backup-operation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (r/'PUBLICATION_HOLD.json').exists():
 atomic(r/'backup-latest-result.json',{'status':'publication_blocked','epoch':time.time(),'local_checkpoints_retained':True});raise SystemExit(3)
result={'started_epoch':time.time()}
try:
 result['models']=tick(c);run(c);result['status']='completed'
except Exception as e:
 result.update(status='failed',error=str(e));raise
finally:
 result['ended_epoch']=time.time();atomic(r/'backup-latest-result.json',result)
