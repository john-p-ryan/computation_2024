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

## Household solver with income shocks (`OG-Core-stoch/`)

* `household_stoch.py`: shock process (Rouwenhorst), the household's
  first-order conditions and budget constraint (OG-Core's, with labor
  productivity e z), inputs by age and type (bequests, transfers,
  remittances, UBI, pensions, taxes), the batched endogenous-grid solver
  (`solve_lifecycle`, `solve_HH`, `solve_all_households_ss`), the
  population-weighted stationary distribution and Euler-error checks
* `household_stoch_kernels.py`: numba kernels (marginal utilities, tax
  functions with analytic derivatives, per-point Newton solvers, monotone
  cubic interpolation, Young's method) with numpy fallbacks
* `docs/household_model.md`: the model, the algorithm, Gate 2 results
  and known issues
* Tests: `hh_test.py` (pointwise functions; replication of OG-Core's
  `SS.inner_loop` for all 7 types), `test_household_stoch_kernels.py`,
  `test_household_stoch_solver.py` (shocks, distribution, accuracy,
  borrowing constraint, failures, backends, economics),
  `test_household_stoch_variants.py` (OG-Core's FOCs under other tax
  functions, analytical MTRs, filers and noncompliance; other pension
  systems). `uv run pytest -m "not local"` is what CI runs (about 4
  minutes on 2 cores once numba has compiled; the first run compiles and
  caches the kernels); `-m local` adds the timing test.

The per-point loops use all cores; set `NUMBA_NUM_THREADS` to limit them.
`OGCORE_STOCH_DISABLE_NUMBA=1` forces the numpy kernels.
