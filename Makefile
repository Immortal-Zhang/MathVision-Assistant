.PHONY: data smoke eval app test clean

data:
	python scripts/make_demo_data.py

smoke:
	python scripts/run_smoke_test.py --backend mock

eval:
	python scripts/run_eval.py --backend mock --top_k 3

app:
	python -m mathvision.app.gradio_app

test:
	pytest

clean:
	rm -rf data/outputs/*.pkl reports/eval_results.csv reports/eval_summary.md .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
