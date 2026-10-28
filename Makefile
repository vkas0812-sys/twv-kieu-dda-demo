PY ?= $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; elif [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
export PYTHONPATH := packages:apps:apps/web

.PHONY: web api test public

web:
	$(PY) -m streamlit run apps/web/streamlit_app.py --server.port 8501

api:
	$(PY) -m uvicorn api.main:app --app-dir apps --port 8000

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q tests/

public:
	npx --yes localtunnel --port 8501
