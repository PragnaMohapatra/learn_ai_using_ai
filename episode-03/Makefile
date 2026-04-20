.PHONY: build test shell help

build:
    docker compose build cli

test:
    docker compose run --rm --entrypoint python cli -m pytest -q

shell:
    docker compose run --rm cli bash

help:
    @echo ""
    @echo "Decode AI Using AI — Episode 3"
    @echo "===================================="
    @echo "  make build   Build the Episode 3 image"
    @echo "  make test    Run the Episode 3 test suite"
    @echo "  make shell   Open a shell inside the Episode 3 container"
    @echo ""