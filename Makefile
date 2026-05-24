.PHONY: install test demo docker

install:
	pip install -e .

test:
	PYTHONPATH=src pytest -q tests/

demo:
	PYTHONPATH=src python scripts/run_demo.py

docker-build:
	docker build -t wpli-pipeline .

docker-test:
	docker run --rm -v $(PWD):/work -w /work --entrypoint pytest wpli-pipeline -q tests/
