#!/usr/bin/env bash
set -euo pipefail

##########################
# 可配置项
##########################
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_NAME="Miniconda3.sh"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="SymmetryML"
PYTHON_VERSION="3.13"

##########################
# 0. 依赖 & tmux（Ubuntu）
##########################
echo "==> 更新 apt 并安装依赖（wget, ca-certificates, tmux）"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y wget ca-certificates tmux

##########################
# 1. 下载并安装 Miniconda（静默）
##########################
echo "==> 下载 Miniconda 安装脚本"
wget -q "${MINICONDA_URL}" -O "${INSTALLER_NAME}"

echo "==> 运行 Miniconda 安装（-b: 批量模式，自动同意）"
bash "${INSTALLER_NAME}" -b -p "${CONDA_DIR}"

##########################
# 2. 初始化 conda（当前会话 + 持久化）
##########################
echo "==> 初始化 conda 到当前会话"
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  # 官方推荐：优先 source conda.sh
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
  # 兜底：使用 hook
  eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
fi

echo "==> 将 conda 初始化写入 shell 配置（持久化）"
# 对 bash / zsh 都进行 init（有就更新，没有就添加），避免不同 shell 下 tmux 拿不到 conda
"${CONDA_DIR}/bin/conda" init bash >/dev/null 2>&1 || true
"${CONDA_DIR}/bin/conda" init zsh  >/dev/null 2>&1 || true

# 可选：不自动激活 base
"${CONDA_DIR}/bin/conda" config --set auto_activate_base false >/dev/null 2>&1 || true

##########################
# 3. 接受官方源 ToS（若命令不可用则忽略）
##########################
echo "==> 接受 Anaconda 官方主源 ToS（若不支持将跳过）"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

##########################
# 4. 创建并激活环境
##########################
echo "==> 创建环境 ${ENV_NAME}（Python ${PYTHON_VERSION}）"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo "==> 激活环境 ${ENV_NAME}"
conda activate "${ENV_NAME}"

##########################
# 5. 使用 pip 安装 Python 包
##########################
echo "==> 使用 pip 安装 comet-ml, torch, numpy, pandas"
pip install -U pip
pip install comet-ml torch numpy pandas

##########################
# 6. 修复 tmux 会话内找不到 conda 的问题
##########################
echo "==> 配置 tmux 让新窗口使用登录 shell，并继承 conda 环境"

TMUX_CONF="${HOME}/.tmux.conf"

# 追加（若不存在）
append_once() {
  local line="$1"
  local file="$2"
  grep -qxF "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

# 保证文件存在
touch "${TMUX_CONF}"

# 1) 让 tmux 使用 bash 并以登录方式启动，这样会加载 /etc/profile 与 ~/.profile（Ubuntu 默认 ~/.profile 会再 source ~/.bashrc）
append_once 'set -g default-shell /bin/bash' "${TMUX_CONF}"
append_once 'set -g default-command "bash -l"' "${TMUX_CONF}"

# 2) 让 tmux 更新 PATH 等环境（含 PATH）
append_once 'set -g update-environment "DISPLAY SSH_ASKPASS SSH_AUTH_SOCK SSH_CONNECTION WINDOWID XAUTHORITY PATH"' "${TMUX_CONF}"

# 3) 将关键变量直接注入 tmux server（即便 tmux server 早于 conda 安装启动）
tmux start-server >/dev/null 2>&1 || true
tmux set-environment -g PATH "${CONDA_DIR}/bin:${PATH}" || true
tmux set-environment -g CONDA_EXE "${CONDA_DIR}/bin/conda" || true
tmux set-environment -g CONDA_SHLVL "0" || true
tmux source-file "${TMUX_CONF}" || true

# 提示：已有的 tmux 窗口/面板是老环境，打开一个新 window/pane 即可生效

##########################
# 7. 清理与验证
##########################
echo "==> 清理安装脚本"
rm -f "${INSTALLER_NAME}"

echo "==> 验证 conda / Python / tmux"
echo -n "conda 版本："; conda --version
echo -n "环境列表："; conda info --envs | grep "${ENV_NAME}" || true
echo -n "Python 版本："; python --version
echo -n "Comet-ML 版本："; python -c "import comet_ml; print(comet_ml.__version__)"
echo -n "tmux 版本："; tmux -V

# 额外：在一个临时 tmux 会话里验证 conda 是否可用（不影响你的会话）
echo "==> 在 tmux 临时会话中验证 conda 可见性（显示成功信息后会自动退出）"
tmux new-session -d -s _conda_check 'bash -lc "conda --version && python --version && echo OK_in_tmux && sleep 1"'
sleep 1
tmux capture-pane -pt _conda_check:0.0 | tail -n 5 || true
tmux kill-session -t _conda_check >/dev/null 2>&1 || true

echo "==> 一切就绪！新开一个 tmux window/pane 即可使用
