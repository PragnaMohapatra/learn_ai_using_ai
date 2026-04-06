.PHONY: build test run-analyze shell help

build:
	docker compose build cli

test:
	docker compose run --rm --entrypoint python cli -m pytest -q

run-analyze:
	docker compose run --rm cli analyze \
		/workspace/episode-01/data/cleaned/customers_clean.csv \
		/workspace/episode-02/data/reports \
		--charts

shell:
	docker compose run --rm cli bash

help:
	@echo ""
	@echo "Decode AI Using AI — Episode 2"
	@echo "===================================="
	@echo "  make build        Build the Episode 2 image"
	@echo "  make test         Run the Episode 2 test suite"
	@echo "  make run-analyze  Reuse Episode 1 cleaned data, render charts, and write reports"
	@echo "  make shell        Open a shell inside the Episode 2 container"
	@echo ""
