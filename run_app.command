#!/bin/zsh
cd "$(dirname "$0")"
if [ -x "../.venv/bin/python" ]; then
  PY="../.venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi
export PYTHONPATH="packages:apps:apps/web"
echo "🚀 Đang khởi động TWV Planning Desk tại http://localhost:8501 ..."
"$PY" -m streamlit run apps/web/streamlit_app.py --server.port 8501
