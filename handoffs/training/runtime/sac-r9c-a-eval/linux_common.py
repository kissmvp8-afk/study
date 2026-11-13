"""Small auditable stdlib helpers; no simulator imports."""
import json, os, hashlib, time
from pathlib import Path

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        file.write('\n'); file.flush(); os.fsync(file.fileno())
    temp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def prepare_args(args):
    args.root = args.root.resolve()
    for name in ('tmp', 'cache', 'cache/cuda'): (args.root/name).mkdir(parents=True, exist_ok=True)
    os.environ.update(PIP_NO_CACHE_DIR='1', TMPDIR=str(args.root/'tmp'), XDG_CACHE_HOME=str(args.root/'cache'), CUDA_CACHE_PATH=str(args.root/'cache/cuda'))
    flags = '--allow-root --/app/extensions/fsWatcherEnabled=false'
    # AppLauncher owns forwarding these Kit flags. No direct SimulationApp CLI injection.
    args.kit_args = (getattr(args, 'kit_args', '') + ' ' + flags).strip()
    return args

class Progress:
    def __init__(self, output):
        self.output=Path(output); self.data={}; self.update('startup')
    def update(self, stage=None, **kwargs):
        if stage is not None and stage != self.data.get('stage'):
            self.data.update(stage=stage, stage_changed_epoch=time.time())
        self.data.update(kwargs, updated_epoch=time.time())
        write_json(self.output/'progress.json', self.data)
