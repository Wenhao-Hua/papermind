.PHONY: install test benchmark dashboard
install:
	python -m pip install -e '.[dev,web,train]'
test:
	python -m pytest tests
benchmark:
	python -m evaluation.eval_retrieval --methods bm25 --max-papers 40 --out evaluation/results/bm25-dev-40.json
dashboard:
	python -m evaluation.dashboard evaluation/results/bm25-dev-40.json
