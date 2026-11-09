"""仅下载/校验/读取已公开证据，不运行包内代码。"""
import concurrent.futures,hashlib,json,urllib.request,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent;D=json.loads((R/'input.json').read_text())
if 'manifest' not in D:
 with urllib.request.urlopen(D['manifest_url'],timeout=60) as s:raw=s.read()
 assert hashlib.sha256(raw).hexdigest()==D['sidecar']['manifest_sha256']
 D['manifest']=json.loads(raw)
def get(a):
 p=R/a['name']
 if not p.exists():
  with urllib.request.urlopen(a['url'],timeout=60) as s,p.open('wb') as f:
   while b:=s.read(1024*1024):f.write(b)
 assert p.stat().st_size==a['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256']
 return p
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:pp=list(ex.map(get,D['parts']))
zpath=R/'raw.zip'
with zpath.open('wb') as f:
 for p in pp:f.write(p.read_bytes())
assert zpath.stat().st_size==D['sidecar']['bytes']
assert hashlib.sha256(zpath.read_bytes()).hexdigest()==D['sidecar']['core_sha256']
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None
 for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
 z.extractall(R/'evidence')
for f in D['manifest']['files']:
 p=R/'evidence'/f['path'];b=p.read_bytes()
 assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],f['path']
report={'commit':D['commit'],'bytes':zpath.stat().st_size,'sha256':D['sidecar']['core_sha256'],'pieces':len(pp),'members_verified':len(D['manifest']['files']),'CRC':'passed'}
(R/'archive-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
