
.PHONY: quality style test

quality:
	isort --check-only .
	black --check .
	flake8 --max-line-length 119 --ignore=E203,W503 --exclude=.venv,venv,.env,env,build,dist .

style:
	isort .
	black .

test:
	pytest -sv ./src/

pip:
	rm -rf build/
	rm -rf dist/
	python -m build
	twine upload dist/* --verbose