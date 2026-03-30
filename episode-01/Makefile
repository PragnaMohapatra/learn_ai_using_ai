.PHONY: build clean run-clean run-clean-fill run-notebook run-all jupyter down help

# --- Build the Docker image ---
build:
	docker compose build

# --- Clean CSV (drop missing rows) ---
run-clean:
	docker compose run --rm cli clean \
		/workspace/data/raw/customers.csv \
		/workspace/data/cleaned/customers_clean.csv \
		--missing drop --json

# --- Clean CSV (fill missing values) ---
run-clean-fill:
	docker compose run --rm cli clean \
		/workspace/data/raw/customers.csv \
		/workspace/data/cleaned/customers_clean.csv \
		--missing fill --json

# --- Run the analysis notebook ---
run-notebook:
	docker compose run --rm cli run-nb \
		/workspace/notebooks/analyze_customers.ipynb \
		--json

# --- Full pipeline: clean then run notebook ---
run-all: run-clean run-notebook

# --- Launch Jupyter Lab for interactive exploration ---
jupyter:
	docker compose up jupyter

# --- Stop all containers ---
down:
	docker compose down

# --- Remove built images ---
clean:
	docker compose down --rmi all --volumes --remove-orphans

# --- Show help ---
help:
	@echo ""
	@echo "Decode AI Using AI — Episode 1"
	@echo "===================================="
	@echo ""
	@echo "  make build          Build the Docker image"
	@echo "  make run-clean      Clean CSV (drop missing rows)"
	@echo "  make run-clean-fill Clean CSV (fill missing values)"
	@echo "  make run-notebook   Execute the analysis notebook"
	@echo "  make run-all        Full pipeline: clean + notebook"
	@echo "  make jupyter        Launch Jupyter Lab (port 8888)"
	@echo "  make down           Stop all containers"
	@echo "  make clean          Remove images and volumes"
	@echo "  make help           Show this help"
	@echo ""
