"""R6 显式清单发布；只创建实验分支/预发布资产，所有回执独立于核心包。"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile
from r7p_backup_ops import atomic,read,sha

REPO='kissmvp8-afk/study'
BRANCH='experiment/sac-r7p-precision-20261109'
TAG='sac-r7p-precision-20261109'
PREFIX='results/tita-sac/sac-r7p-precision-20261109/'
PATTERNS=[rb'gh[pousr]_[A-Za-z0-9]{20,}',rb'github_pat_[A-Za-z0-9_]{20,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'https?://[^/\s:@]+:[^/\s@]+@',rb'AKIA[0-9A-Z]{16}',rb'(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[=:]\s*[\"\'][A-Za-z0-9_+/=.-]{16,}[\"\']']
FORBIDDEN=('.git','.ssh','.aws','.env','credentials','credential','id_rsa','id_ed25519','token','tokens')

def safe_name(name):
    p=Path(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or any(x.lower() in FORBIDDEN for x in p.parts):raise ValueError('Unsafe publication path: '+name)

def scan_stream(stream,label):
    tail=b''
    while True:
        chunk=stream.read(1024*1024)
        if not chunk:break
        data=tail+chunk
        if any(re.search(p,data) for p in PATTERNS):raise ValueError('Possible credential in staging: '+label)
        tail=data[-4096:]

def scan_file(path):
    with Path(path).open('rb') as f:scan_stream(f,Path(path).name)
    # Scan actual expanded ZIP bytes, never infer clean from compressed-byte scanning.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise ValueError('ZIP CRC failure')
            for item in z.infolist():
                safe_name(item.filename)
                if not item.is_dir():
                    with z.open(item) as f:scan_stream(f,item.filename)

def validate_plan(spec):
    if spec['repository']!=REPO or spec['branch']!=BRANCH or spec['tag']!=TAG:raise ValueError('Only isolated R7P destination allowed')
    if not re.fullmatch('[0-9a-f]{40}',spec['base_commit']):raise ValueError('Pinned base commit required')
    root=Path(spec['staging_root']).resolve()
    files=spec['files'];seen=set();sources=set()
    if not files:raise ValueError('Empty explicit allowlist')
    for row in files:
        safe_name(row['source']);p=root/row['source'];resolved=p.resolve();resolved.relative_to(root)
        if p.is_symlink() or not p.is_file():raise ValueError('Regular staging file required')
        if row['bytes']!=p.stat().st_size or row['sha256']!=sha(p):raise ValueError('Staging differs from fixed manifest')
        if row['source'] in sources:raise ValueError('Duplicate source');
        sources.add(row['source'])
        kind=row['kind']
        if kind=='git':
            dest=row['destination'];safe_name(dest)
            if not dest.startswith(PREFIX) or p.stat().st_size>24000000:raise ValueError('Git result destination/size restriction')
        elif kind in ('model','resume','evidence'):
            dest=row['asset_name'];safe_name(dest)
            if Path(row['source']).name!=dest or '/' in dest or row['sha256'][:16] not in dest:raise ValueError('Asset name must contain digest and be flat')
            if kind=='evidence' and p.stat().st_size>24000000:raise ValueError('Evidence asset >24MB')
            if kind in ('model','resume') and (p.suffix!='.pt' or not row.get('training_identity')):raise ValueError('Explicit model identity required')
        else:raise ValueError('Unknown allowlist category')
        identity=(kind=='git',dest)
        if identity in seen:raise ValueError('Duplicate remote destination')
        seen.add(identity);scan_file(p)
    finals=[r for r in files if r.get('final_candidate')]
    if finals and (len(finals)!=2 or any(r['kind']!='model' for r in finals)):
        raise ValueError('Final candidate verification requires the exact pair of model assets')
    # Split archives require the combined core.zip to be scanned and hash-verified too.
    for archive in spec.get('split_archives',[]):
        safe_name(archive['source']);p=root/archive['source'];p.resolve().relative_to(root)
        if p.is_symlink() or sha(p)!=archive['sha256']:raise ValueError('Combined archive SHA mismatch')
        scan_file(p);joined=hashlib.sha256()
        for part in archive['parts']:
            safe_name(part)
            if part not in sources:raise ValueError('Archive piece absent from upload allowlist')
            with (root/part).open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):joined.update(chunk)
        if joined.hexdigest()!=archive['sha256']:raise ValueError('Split archive bytes mismatch')
    split_sources={p for a in spec.get('split_archives',[]) for p in a['parts']}
    if any('.part' in r['source'] and r['source'] not in split_sources for r in files):raise ValueError('Unvalidated split archive')
    return root

def sanitized_cli_failure(returncode,stderr='',stdout='',timed_out=False):
    """只提取固定枚举与有界HTTP数值；从不返回原文、URL或命令参数。"""
    raw='\n'.join(x for x in (stderr,stdout) if isinstance(x,str))
    statuses=sorted(set(int(m) for m in re.findall(r'(?i)\bHTTP(?:/[0-9.]+)?\s+([45][0-9]{2})\b',raw)))[:5]
    retry=None
    match=re.search(r'(?im)^\s*retry-after\s*:\s*([0-9]{1,7})\s*$',raw)
    if match and int(match[1])<=604800:retry=int(match[1])
    lower=raw.lower()
    limited=429 in statuses or any(x in lower for x in ('rate limit exceeded','rate-limit exceeded','secondary rate limit','primary rate limit','api rate limit'))
    if timed_out:category='timeout'
    elif limited:category='rate_limit'
    elif 401 in statuses:category='authentication_required'
    elif 403 in statuses:category='access_denied'
    elif 404 in statuses:category='not_found_or_not_visible'
    elif 409 in statuses or 422 in statuses:category='conflict_or_validation'
    elif any(x>=500 for x in statuses):category='remote_service_error'
    elif statuses:category='http_error'
    elif any(x in lower for x in ('connection refused','connection reset','network is unreachable','no such host','could not resolve','tls handshake timeout')):category='connection_error'
    else:category='unclassified_cli_failure'
    return dict(exit_code=returncode,category=category,http_statuses=statuses,retry_after_seconds=retry,
                raw_output_retained=False,retry_policy_changed=False)

class GitHub:
    def __init__(self,gh,scratch):self.gh=str(gh);self.scratch=Path(scratch)
    def call(self,args):
        try:
            result=subprocess.run([self.gh,*args],capture_output=True,text=True,timeout=300)
        except subprocess.TimeoutExpired:
            safe=sanitized_cli_failure(None,timed_out=True)
            atomic(self.scratch/('cli-failure-%d.sidecar.json'%time.time_ns()),safe)
            raise RuntimeError('GitHub command failed: '+json.dumps(safe,sort_keys=True)) from None
        if result.returncode:
            safe=sanitized_cli_failure(result.returncode,result.stderr,result.stdout)
            atomic(self.scratch/('cli-failure-%d.sidecar.json'%time.time_ns()),safe)
            raise RuntimeError('GitHub command failed: '+json.dumps(safe,sort_keys=True))
        return result.stdout
    def api(self,path,method='GET',body=None):
        args=['api',path,'--method',method]
        if body is not None:
            temp=self.scratch/('request-%d.json'%time.time_ns());atomic(temp,body);args+=['--input',str(temp)]
            try:return json.loads(self.call(args))
            finally:temp.unlink(missing_ok=True)
        return json.loads(self.call(args))
    def branches(self):return self.api('repos/'+REPO+'/git/matching-refs/heads/'+BRANCH)
    def head(self):
        found=[r for r in self.branches() if r['ref']=='refs/heads/'+BRANCH]
        if len(found)>1:raise ValueError('Ambiguous remote branch')
        return found[0]['object']['sha'] if found else None

def publish(spec_path,gh,receipt_dir):
    spec=read(spec_path);root=validate_plan(spec);receipts=Path(receipt_dir).resolve()
    if receipts==root or receipts.is_relative_to(root):raise ValueError('Receipts must be outside frozen staging')
    receipts.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(receipts/'publish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    api=GitHub(gh,receipts);run=receipts/('publish-%d.json'%time.time_ns())
    receipt=dict(plan_sha256=sha(spec_path),started_epoch=time.time(),repository=REPO,branch=BRANCH,tag=TAG,status='active',files=[],credential_scan='passed_actual_staging')
    atomic(run,receipt)
    try:
        head=api.head()
        if head is None:
            api.api('repos/'+REPO+'/git/commits/'+spec['base_commit'])
            try:api.api('repos/'+REPO+'/git/refs','POST',{'ref':'refs/heads/'+BRANCH,'sha':spec['base_commit']})
            except Exception:
                if api.head()!=spec['base_commit']:raise
            head=api.head()
        # Refuse stale expected branch, never force-push or modify main.
        expected=spec.get('expected_head')
        if expected and head!=expected:raise ValueError('Remote head changed; regenerate plan after reconciliation')
        commit=api.api('repos/'+REPO+'/git/commits/'+head);base_tree=commit['tree']['sha'];tree=[]
        for row in spec['files']:
            if row['kind']!='git':continue
            p=root/row['source'];data=p.read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Staging changed before blob creation')
            scan_stream(io.BytesIO(data),row['source'])
            blob=api.api('repos/'+REPO+'/git/blobs','POST',dict(content=base64.b64encode(data).decode(),encoding='base64'))
            tree.append(dict(path=row['destination'],mode='100644',type='blob',sha=blob['sha']))
        if tree:
            new_tree=api.api('repos/'+REPO+'/git/trees','POST',dict(base_tree=base_tree,tree=tree))['sha']
            if new_tree!=base_tree:
                new_commit=api.api('repos/'+REPO+'/git/commits','POST',dict(message='Record fixed R6 evidence '+sha(spec_path)[:16],tree=new_tree,parents=[head]))['sha']
                receipt.update(pending_commit=new_commit,parent_commit=head);atomic(run,receipt)
                # Coordinate again immediately before ref update.
                if api.head()!=head:raise ValueError('Remote branch moved during staging')
                try:api.api('repos/'+REPO+'/git/refs/heads/'+BRANCH,'PATCH',dict(sha=new_commit,force=False))
                except Exception:
                    if api.head()!=new_commit:raise
                head=api.head()
                if head!=new_commit:raise ValueError('Commit publication not verified')
        receipt['verified_commit']=head;atomic(run,receipt)
        assets=[r for r in spec['files'] if r['kind']!='git']
        if assets:
            releases=api.api('repos/'+REPO+'/releases?per_page=100');matches=[r for r in releases if r['tag_name']==TAG]
            if not matches:
                receipt['release_create_pending']=True;atomic(run,receipt)
                try:rel=api.api('repos/'+REPO+'/releases','POST',dict(tag_name=TAG,target_commitish=BRANCH,name='SAC R7P precision pilot evidence',body='Experimental R7P artifacts. Not control acceptance or real-world validation.',draft=False,prerelease=True))
                except Exception:
                    rel=api.api('repos/'+REPO+'/releases/tags/'+TAG)
            else:rel=matches[0]
            if rel['target_commitish']!=BRANCH or rel['prerelease'] is not True or rel['draft']:raise ValueError('Existing release does not match isolated R6 contract')
            def listing():
                return [x for page in json.loads(api.call(['api','--paginate','--slurp','repos/'+REPO+'/releases/'+str(rel['id'])+'/assets'])) for x in page]
            for row in assets:
                if sha(root/row['source'])!=row['sha256']:raise ValueError('Asset changed before upload')
                scan_file(root/row['source'])
                existing=[x for x in listing() if x['name']==row['asset_name']]
                if not existing:
                    pending=dict(name=row['asset_name'],status='upload_pending',sha256=row['sha256'],bytes=row['bytes']);receipt['files'].append(pending);atomic(run,receipt)
                    try:api.call(['release','upload',TAG,str(root/row['source']),'--repo',REPO])
                    except Exception:
                        # Upload outcome may be unknown: list once, never repeat upload here.
                        pass
                    existing=[x for x in listing() if x['name']==row['asset_name']]
                if len(existing)!=1 or existing[0].get('state')!='uploaded' or existing[0]['size']!=row['bytes'] or existing[0].get('digest')!='sha256:'+row['sha256']:
                    raise ValueError('Asset missing/conflict/unverified: '+row['asset_name'])
                downloaded=False
                if row.get('final_candidate'):
                    downloaded_path=receipts/('download-'+row['asset_name'])
                    with downloaded_path.open('wb') as out:
                        result=subprocess.run([str(gh),'api','repos/'+REPO+'/releases/assets/'+str(existing[0]['id']),'-H','Accept: application/octet-stream'],stdout=out,stderr=subprocess.PIPE,timeout=300)
                    if result.returncode or downloaded_path.stat().st_size!=row['bytes'] or sha(downloaded_path)!=row['sha256']:
                        raise ValueError('Final candidate actual re-download verification failed')
                    downloaded=True
                receipt['files'].append(dict(name=row['asset_name'],status='actual_redownload_verified' if downloaded else 'remote_digest_size_verified',asset_id=existing[0]['id'],sha256=row['sha256'],bytes=row['bytes'],actual_redownload=downloaded));atomic(run,receipt)
        receipt.update(status='remote_verified',completed_epoch=time.time(),commit_url='https://github.com/'+REPO+'/commit/'+head)
        atomic(run,receipt);return receipt
    except Exception as exc:
        receipt.update(status='blocked_or_uncertain',error=str(exc),completed_epoch=time.time());atomic(run,receipt);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--gh',required=True);p.add_argument('--receipts',required=True);a=p.parse_args();publish(a.plan,a.gh,a.receipts)
