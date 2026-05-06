.PHONY: test lint format docs clean

test:
	pytest tests/

lint:
	black --check . && flake8 . && isort --check .

format:
	black . && isort .

docs:
	cd docs && make html

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
