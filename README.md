# computation_2024
Code repository for working towards updating OG Core with income shocks

## Environment

Python 3.12 with [uv](https://docs.astral.sh/uv/). `ogcore` is pinned to
0.20.1 in `pyproject.toml`; `uv.lock` pins everything else.

```
uv sync                                  # create .venv from uv.lock
uv run pytest                            # prototype suite in OG-Core-stoch/
uv run ruff format . && uv run ruff check .
```

`environment.yml` is a conda fallback with the same pins.
