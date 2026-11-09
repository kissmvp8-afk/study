"""下载固定证据，不执行任何归档内代码。"""
import concurrent.futures,hashlib,json,urllib.request,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
D=json.loads((R/'input.json').read_text())
def get(a):
    p=R/a['name']
    if not p.exists():
        with urllib.request.urlopen(a['url'],timeout=60) as s,p.open('wb') as f:
            while b:=s.read(1024*1024):f.write(b)
    assert p.stat().st_size==a['bytes'],p
    assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'],p
    return {'name':p.name,'bytes':p.stat().st_size,'sha256':a['sha256']}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    records=list(ex.map(get,D['parts']+D['models']))
zpath=R/'raw-evidence.zip'
with zpath.open('wb') as dst:
    for a in D['parts']:
        with (R/a['name']).open('rb') as f:
            while b:=f.read(1024*1024):dst.write(b)
print('archive',zpath.stat().st_size,hashlib.sha256(zpath.read_bytes()).hexdigest(),flush=True)
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    names=z.namelist()
    (R/'zip-members.json').write_text(json.dumps([{'name':i.filename,'bytes':i.file_size} for i in z.infolist()]))
    for name in names:
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts
    z.extractall(R/'evidence')
(R/'download-verification.json').write_text(json.dumps({'input_commit':D['input_commit'],'CRC':'passed','archive_sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),'files':records},indent=2))
print('download and CRC completed',len(records),'assets',len(names),'archive entries',flush=True)
