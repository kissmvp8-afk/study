"""恢复已安装Kit官方命令注册；不替换命令实现、不更改debug_vis/物理设置。"""
import importlib,json,re
from pathlib import Path
REQUESTS={'CreatePreviewSurfaceMaterialPrim':('CreatePreviewSurfaceMaterialPrim','CreatePreviewSurfaceMaterialPrimCommand'),'ChangePropertyCommand':('ChangePropertyCommand',)}
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def class_identity(cls):return None if cls is None else cls.__module__+'.'+cls.__qualname__
def command_class(commands,name):
 try:return commands.get_command_class(name)
 except Exception as exc:raise RuntimeError('Command lookup failed/ambiguous; refuse destructive re-registration: '+name+': '+str(exc))
def repair(manifest_path):
 import omni.kit.app,omni.kit.commands
 commands=omni.kit.commands;manager=omni.kit.app.get_app().get_extension_manager()
 providers=json.loads(Path(manifest_path).read_text())['providers'];records=[]
 for requested,candidates in REQUESTS.items():
  found=[]
  for p in providers:
   for cls in candidates:
    if re.search(r'class\s+'+re.escape(cls)+r'\b',p['source']):found.append((p,cls))
  require(len(found)==1,'Need exactly one installed source provider for '+requested)
  provider,cls_name=found[0];file=Path(provider['file']).resolve();root=Path(provider['extension_root']).resolve()
  require(file.is_file() and file.read_text()==provider['source'],'Provider source changed since inspection')
  relative=file.relative_to(root).with_suffix('');parts=list(relative.parts)
  require('omni' in parts,'Provider is not an installed omni module')
  module_name='.'.join(parts[parts.index('omni'):]);module_name=module_name.removesuffix('.__init__')
  extension_name=re.split(r'-(?=\d)',root.name,maxsplit=1)[0]
  before=command_class(commands,requested);before_enabled=manager.is_extension_enabled(extension_name)
  actions=[]
  if before is None:
   if not before_enabled:
    require(manager.set_extension_enabled_immediate(extension_name,True) is not False,'Cannot enable installed provider '+extension_name);actions.append('enable_installed_extension_in_process')
   module=importlib.import_module(module_name);provided=getattr(module,cls_name);actions.append('import_official_provider_module')
   current=command_class(commands,requested)
   if current is None:
    commands.register(provided);actions.append('register_exact_official_class_once')
   after=command_class(commands,requested)
   require(after is provided,'Resolved command differs from exact installed class')
  else:after=before
  require(after is not None,'Command remains unavailable '+requested)
  records.append({'command':requested,'provider_file':str(file),'module':module_name,'extension':extension_name,'extension_root':str(root),'enabled_before':before_enabled,'enabled_after':manager.is_extension_enabled(extension_name),'before':class_identity(before),'after':class_identity(after),'actions':actions})
 return {'status':'passed','scope':'process-local installed official command registration only','providers':records,'physics_or_debug_configuration_changed':False}
