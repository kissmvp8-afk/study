"""R7P独立文件/时间工具；没有R6旧预算或预检门控。"""
import datetime as dt
import hashlib,json,os
from pathlib import Path
PLAN='SAC-R7P-PRECISION-PILOT-20261109'
PARENT='cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3'
ARMS=(('P0',1401),('P1',1401),('P0',1402),('P1',1402))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 with tmp.open('w') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def epoch(s):
 x=dt.datetime.fromisoformat(s.replace('Z','+00:00'))
 if x.tzinfo is None:raise ValueError('Explicit timezone required')
 return x.timestamp()
def iso(t):return dt.datetime.fromtimestamp(t,dt.timezone.utc).isoformat()
def verify(rows):
 for row in rows:
  p=Path(row['path'])
  if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Frozen input changed '+str(p))
def file_record(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
