#!/usr/bin/env bash
# 在已有 NVIDIA 驱动的新 Ubuntu 4090 实例部署隔离环境；不启动正式训练。
set -Eeuo pipefail
if [[ ${1:-} == --print-plan ]]; then
    cat <<'PLAN'
Ubuntu 22.04/24.04 x86_64 + RTX4090, retain host NVIDIA driver.
Python 3.11; torch2.7.0+cu128; torchvision0.22.0+cu128; isaacsim5.1.0.
IsaacLab 3c6e67bb5c7ada942a6d1884ab69338f57596f77 (recorded v2.3.0).
DDT_Lab 701a49c6e78dfb93a8b30145f0e7723b22921c0b.
ddt_ros2_control e7b32d9746df900680641ff2845ffb0b5ffe99ed.
study 890cacb9d14519bb028c4d51f11d9c4d401d8165; create experiment branch later.
All environments, Python binaries, downloads and task caches under selected data directory.
Install only IsaacLab core/assets/tasks/rl packages; do not install every RL framework.
Run CUDA math, headless scene and 128-Tita collection smoke checks.
Does not implement/validate full SAC-R9C learner, supervisor or complete resume.
PLAN
    exit 0
fi
if [[ ${1:-} == --help ]]; then
    printf '%s\n' 'Usage: bash setup_4090.sh /actual/data/tita-r9c' 'Existing working stack: audit/reuse it instead of installing again.'
    exit 0
fi
die() { printf '[BLOCKED] %s\n' "$*" >&2; exit 2; }
[[ $(uname -m) == x86_64 ]] || die 'This installer is for x86_64.'
[[ -r /etc/os-release ]] || die 'Missing /etc/os-release.'
# shellcheck disable=SC1091
source /etc/os-release
[[ $ID == ubuntu && ($VERSION_ID == 22.04 || $VERSION_ID == 24.04) ]] || die 'Use the Ubuntu22.04/24.04 instance specified by this task.'
command -v nvidia-smi >/dev/null || die 'NVIDIA driver unavailable: enable the platform GPU/driver; do not install a kernel driver inside the instance.'
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 4090' || die 'No RTX4090 detected.'
SETUP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -f $SETUP_DIR/smoke_runtime.py && -f $SETUP_DIR/constraints.txt ]] || die 'Download the complete fixed-version handoff folder first.'
[[ $# == 1 ]] || die 'Pass the real data-directory project path, e.g. /root/rivermind-data/tita-r9c.'
TITA_PROJECT_ROOT=$(realpath -m -- "$1")
[[ $TITA_PROJECT_ROOT != / && $TITA_PROJECT_ROOT != /root && $TITA_PROJECT_ROOT != /usr ]] || die 'Choose a dedicated project directory on the data disk.'
probe=$TITA_PROJECT_ROOT
while [[ ! -e $probe ]]; do probe=$(dirname -- "$probe"); done
free_kib=$(df -Pk -- "$probe" | awk 'NR==2 {print $4}')
(( free_kib >= 40*1024*1024 )) || die 'New installation needs at least40GiB available here; inspect the actual data mount.'
mkdir -p "$TITA_PROJECT_ROOT"/{cache,tmp,tools,logs,runs,models,python,runtime}
chmod 700 "$TITA_PROJECT_ROOT/runtime"
exec 9>"$TITA_PROJECT_ROOT/setup.lock"
flock -n 9 || die 'Another setup process holds this task lock.'
exec > >(tee -a "$TITA_PROJECT_ROOT/logs/setup.log") 2>&1
trap 'rc=$?; printf "SETUP_FAILED rc=%s line=%s; preserve logs and partial environment\n" "$rc" "$LINENO" >&2; exit "$rc"' ERR
export TMPDIR="$TITA_PROJECT_ROOT/tmp" XDG_CACHE_HOME="$TITA_PROJECT_ROOT/cache" CUDA_CACHE_PATH="$TITA_PROJECT_ROOT/cache/cuda"
export UV_CACHE_DIR="$TITA_PROJECT_ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$TITA_PROJECT_ROOT/python"
export PIP_NO_CACHE_DIR=1 UV_LINK_MODE=copy
mkdir -p "$CUDA_CACHE_PATH" "$UV_CACHE_DIR"
if (( EUID == 0 )); then APT=(env DEBIAN_FRONTEND=noninteractive apt-get); else
    command -v sudo >/dev/null && sudo -n true || die 'Root or existing passwordless sudo is required for the listed user-space libraries.'
    APT=(sudo -n env DEBIAN_FRONTEND=noninteractive apt-get)
fi
"${APT[@]}" update
GLIB_PACKAGE=libglib2.0-0
[[ $VERSION_ID != 24.04 ]] || GLIB_PACKAGE=libglib2.0-0t64
"${APT[@]}" install -y --no-install-recommends git git-lfs gh curl ca-certificates unzip zip \
    build-essential cmake pkg-config python3-venv python3-pip tmux ffmpeg zstd \
    "$GLIB_PACKAGE" libgl1 libegl1 libx11-6 libxext6 libxi6 libxrender1 libxtst6 \
    libxrandr2 libxinerama1 libxcursor1 libxkbcommon0 libvulkan1 vulkan-tools libgomp1 \
    libx11-xcb1 libxcb1 libxcb-xinerama0 libxkbcommon-x11-0 libsm6 libice6 \
    libnss3 libnspr4 libfontconfig1 libfreetype6 libdbus-1-3 libgbm1 libdrm2
python3 -m venv "$TITA_PROJECT_ROOT/tools/installer-venv"
INSTALL_PY="$TITA_PROJECT_ROOT/tools/installer-venv/bin/python"
"$INSTALL_PY" -m pip install --disable-pip-version-check --retries 2 --timeout 60 uv
UV="$TITA_PROJECT_ROOT/tools/installer-venv/bin/uv"
"$UV" --version
VENV="$TITA_PROJECT_ROOT/.venv"
if [[ ! -e $VENV ]]; then "$UV" venv --python 3.11 --seed "$VENV"; fi
PY="$VENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2]==(3,11), sys.version'
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
# 不继承另一个conda环境的Python选择；不删除其目录。
unset CONDA_PREFIX
"$PY" -m pip install --upgrade pip 'setuptools<81' wheel
cp -- "$SETUP_DIR/constraints.txt" "$TITA_PROJECT_ROOT/constraints.txt"
export PIP_CONSTRAINT="$TITA_PROJECT_ROOT/constraints.txt"
"$PY" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# 使用已成功采用的all组件；不额外下载庞大的extscache全集。
"$PY" -m pip install 'isaacsim[all]==5.1.0' --extra-index-url https://pypi.nvidia.com
clone_pinned() {
    local url=$1 target=$2 revision=$3
    if [[ ! -e $target ]]; then git clone --filter=blob:none "$url" "$target"; fi
    [[ -d $target/.git ]] || die "Not an independent git checkout: $target"
    [[ -z $(git -C "$target" status --porcelain --untracked-files=no) ]] || die "Preserve existing modified checkout: $target"
    [[ $(git -C "$target" remote get-url origin) == "$url" ]] || die "Unexpected repository at $target"
    git -C "$target" fetch origin "$revision"
    git -C "$target" checkout --detach "$revision"
    [[ $(git -C "$target" rev-parse HEAD) == "$revision" ]] || die "Revision mismatch: $target"
}
clone_pinned https://github.com/isaac-sim/IsaacLab.git "$TITA_PROJECT_ROOT/IsaacLab" 3c6e67bb5c7ada942a6d1884ab69338f57596f77
clone_pinned https://github.com/DDTRobot/DDT_Lab.git "$TITA_PROJECT_ROOT/DDT_Lab" 701a49c6e78dfb93a8b30145f0e7723b22921c0b
clone_pinned https://github.com/DDTRobot/ddt_ros2_control.git "$TITA_PROJECT_ROOT/DDT_Lab/ddt_ros2_control" e7b32d9746df900680641ff2845ffb0b5ffe99ed
git -C "$TITA_PROJECT_ROOT/DDT_Lab/ddt_ros2_control" lfs install --local
git -C "$TITA_PROJECT_ROOT/DDT_Lab/ddt_ros2_control" lfs pull
clone_pinned https://github.com/kissmvp8-afk/study.git "$TITA_PROJECT_ROOT/study" 890cacb9d14519bb028c4d51f11d9c4d401d8165
for package in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
    "$PY" -m pip install -e "$TITA_PROJECT_ROOT/IsaacLab/source/$package"
done
"$PY" -m pip install rsl-rl-lib==3.0.1 tensordict==0.14.3 h5py==3.16.0 \
    psutil matplotlib tensorboard gymnasium pyyaml imageio imageio-ffmpeg zstandard
"$PY" -m pip install -e "$TITA_PROJECT_ROOT/DDT_Lab/source/ddt_lab"
urdf="$TITA_PROJECT_ROOT/DDT_Lab/ddt_ros2_control/urdfs/tita_description/urdf/robot.urdf"
[[ $(sha256sum -- "$urdf" | awk '{print $1}') == 0941b83136cd5c8c89bbb33165bff515a3a988d33319b67c8b06bfb9c71350ce ]] || die 'Tita URDF SHA mismatch.'
"$PY" -m pip check
"$PY" -m pip freeze > "$TITA_PROJECT_ROOT/logs/pip-freeze.txt"
"$PY" - <<'PY'
import json,torch,platform
assert torch.__version__=='2.7.0+cu128',torch.__version__
assert torch.cuda.is_available()
assert '4090' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
x=torch.randn(512,512,device='cuda:0'); y=x@x.T
assert torch.isfinite(y).all().item()
torch.cuda.synchronize()
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'CUDA_matmul_passed':True}))
PY
# 导出任务级激活脚本，Python引用repr保证带空格路径的shell安全引用。
"$PY" - "$TITA_PROJECT_ROOT" <<'PY'
import sys,shlex
from pathlib import Path
r=Path(sys.argv[1]);q=shlex.quote
lines=['# Source this in the task shell; do not change HOME.']
for name,path in {'TITA_PROJECT_ROOT':r,'VIRTUAL_ENV':r/'.venv','TMPDIR':r/'tmp','XDG_CACHE_HOME':r/'cache','CUDA_CACHE_PATH':r/'cache/cuda','XDG_RUNTIME_DIR':r/'runtime','UV_CACHE_DIR':r/'cache/uv','UV_PYTHON_INSTALL_DIR':r/'python','PIP_CONSTRAINT':r/'constraints.txt'}.items():lines.append(f'export {name}={q(str(path))}')
lines+=['export PATH="$VIRTUAL_ENV/bin:$PATH"','export PIP_NO_CACHE_DIR=1','export OMNI_KIT_ACCEPT_EULA=YES']
(r/'activate_task.sh').write_text('\n'.join(lines)+'\n')
PY
# shellcheck disable=SC1090
source "$TITA_PROJECT_ROOT/activate_task.sh"
vulkaninfo --summary > "$TITA_PROJECT_ROOT/logs/vulkan-summary.txt" 2>&1 || printf '%s\n' 'Vulkan probe did not finish cleanly; real Isaac smoke must still pass.'
timeout --signal=TERM 1200 "$PY" "$SETUP_DIR/smoke_runtime.py" --mode scene --output "$TITA_PROJECT_ROOT/logs/scene-smoke.json"
timeout --signal=TERM 1200 "$PY" "$SETUP_DIR/smoke_runtime.py" --mode tita --num-envs 128 --steps 32 --output "$TITA_PROJECT_ROOT/logs/tita-smoke.json"
df -h -- "$TITA_PROJECT_ROOT" / > "$TITA_PROJECT_ROOT/logs/disk-after-setup.txt"
printf 'ENVIRONMENT_SMOKE_PASSED; next: full SAC learner, serial/parallel and resume acceptance. Root=%s\n' "$TITA_PROJECT_ROOT"
