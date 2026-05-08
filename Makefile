.PHONY: test lint format type docs clean

test:
	pytest tests/

lint:
	ruff check . && ruff format --check .

check: format lint type

format:
	ruff format . && ruff check --fix .

type:
	dmypy run -- ise/

docs:
	cd docs && make html

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
