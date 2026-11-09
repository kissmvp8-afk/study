"""通过环境变量绑定原冻结入口与提供者清单，在AppLauncher完成后补注册再原样运行。"""
import functools,hashlib,json,os,runpy,sys
from pathlib import Path
original=Path(os.environ['R6_ORIGINAL_EVALUATOR']).resolve();expected=os.environ['R6_ORIGINAL_EVALUATOR_SHA256']
assert hashlib.sha256(original.read_bytes()).hexdigest()==expected
manifest=Path(os.environ['R6_KIT_PROVIDER_MANIFEST']).resolve();expected_manifest=os.environ['R6_KIT_PROVIDER_MANIFEST_SHA256']
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected_manifest
out=Path(sys.argv[sys.argv.index('--output_dir')+1]);out.mkdir(parents=True,exist_ok=True)
from isaaclab.app import AppLauncher
init=AppLauncher.__init__
@functools.wraps(init)
def initialized(self,*args,**kwargs):
 init(self,*args,**kwargs)
 from kit_command_repair import repair
 report=repair(manifest)
 report.update(original_evaluator_sha256=expected,provider_manifest_sha256=expected_manifest,wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with (out/'kit-command-repair.json').open('x') as file:json.dump(report,file,indent=2)
AppLauncher.__init__=initialized
sys.path.insert(0,str(original.parent));sys.argv[0]=str(original)
runpy.run_path(str(original),run_name='__main__')
