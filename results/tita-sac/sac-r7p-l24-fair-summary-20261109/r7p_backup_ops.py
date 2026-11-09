"""R7P 备份所需的最小原子JSON和摘要工具。"""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

PLAN='SAC-R7P-PRECISION-PILOT-20261109'
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
