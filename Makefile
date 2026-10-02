.PHONY: setup data build run figures report fixture dbt-fixture test lint format all clean

RUN := uv run delay-contagion

setup:            ## install the locked environment
	uv sync --locked

data:             ## download + cache the latest 12 BTS months (~400 MB zip, ~80 MB parquet)
	$(RUN) data

build:            ## dbt build on the full data (models + data tests)
	$(RUN) build

run: build        ## full pipeline: dbt build, models, LP, result tables, figures
	$(RUN) analyze
	$(RUN) figures

figures:          ## re-render figures from results/
	$(RUN) figures

report:           ## render site/index.html from results/
	$(RUN) report

fixture:          ## regenerate the committed CI fixture from the warehouse
	$(RUN) fixture

dbt-fixture:      ## dbt build on the committed fixture (what CI runs)
	$(RUN) build --fixture

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

all: data run report

clean:            ## remove the warehouse and dbt artifacts (keeps downloaded data)
	rm -rf data/warehouse.duckdb data/fixture.duckdb dbt/target dbt/logs
