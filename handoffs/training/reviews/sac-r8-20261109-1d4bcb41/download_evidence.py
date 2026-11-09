"""只下载、验摘要并解压指定R8证据，不执行归档内脚本或启动训练。"""
import concurrent.futures
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

B=Path('/workspace/sac-r8-review');v=json.loads((B/'download-inputs.json').read_text())
def fetch(a):
    p=B/a['name'];h=hashlib.sha256();total=0
    with urllib.request.urlopen(a['browser_download_url'],timeout=90) as r, p.open('wb') as f:
        while True:
            block=r.read(1024*1024)
            if not block:break
            h.update(block);total+=len(block);f.write(block)
    assert total==a['size'] and a['digest']=='sha256:'+h.hexdigest(),a['name']
    return dict(name=a['name'],sha256=h.hexdigest(),bytes=total,path=str(p),actual_downloaded=True)

assets=v['L2_archive_parts']+v['L48_models']+v['paired1502_resumes']+v['L0_models']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    receipts=list(ex.map(fetch,assets))
(B/'download-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
parts=sorted(v['L2_archive_parts'],key=lambda a:a['name'].split('.part')[-1]);dest=B/'L2-evidence.zip';h=hashlib.sha256()
with dest.open('wb') as f:
    for a in parts:
        with (B/a['name']).open('rb') as r:
            while True:
                block=r.read(1024*1024)
                if not block:break
                h.update(block);f.write(block)
assert h.hexdigest()==v['L2_verification']['core_sha256']
out=B/'L2-evidence';out.mkdir(exist_ok=True)
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(out)
for f in v['L2_manifest']['files']:
    p=out/f['path'];assert p.is_file() and p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],p
receipt=dict(L2_fixed_commit='d0343a6107a6ca7ea875a519f95027c2ee387e4d',L2_archive_sha256=h.hexdigest(),L2_zip_CRC=True,L2_files_SHA_size_verified=len(v['L2_manifest']['files']),actual_asset_downloads=len(receipts),L48_raw_evaluations='not_yet_available')
(B/'archive-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
