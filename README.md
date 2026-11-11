# Complex Runge–Kutta methods of orders five to eight

Code and data of the paper

> V. Rybin, I. Babkin, A. Karimov, S. S. Muni, D. Butusov. *Complex Explicit Runge–Kutta Methods
> with Real Projection.* Submitted to Mathematics (MDPI).

The methods are explicit Runge–Kutta methods with complex coefficients applied to a real system
y' = f(t, y); after every step only the real part of the result is kept. With this projection, orders
five, six and seven are reached with as many stages as the order and order eight with nine stages.
The repository contains all tableaux with 45 digits, the verification of the order conditions and of
the structure of the solution sets (ranks, the simplifying assumption D(1), the interval Krawczyk test,
the confluent eighth-order scheme L), the minimization of the error constants, the embedded pairs, all
fixed- and variable-step experiments, the C++ timing benchmark, the latency model, and the scripts that
generate every table and figure of the paper.

The complex Cash–Karp pairs CCK6 and CCK7 are the subject of a companion paper and are not included.
Some stored results in `order56/` also cover four further CRK5 and CRK6 tableaux that were studied during
the work but are not reported in the paper; their coefficients are not included, and the scripts of
`paper1/` skip them.

## Layout

| Path | Contents |
|---|---|
| `order56/code/base/` | orders five and six: order conditions and integrator in multiple precision (`crk.py`), tableaux, test problems, measurements, optimization, stability; data in `data/` |
| `order56/code/r1/` | orders five and six: structure of the solution sets, Krawczyk tests, newer real pairs, robustness checks, N-body and BRUSS-2D, latency model, extrapolation methods; `r1common.py` sets all paths |
| `order78/code/`, `order78/data/` | orders seven and eight: the same pipeline, the confluent limit, fixed-point rounding, Lorenz-96 |
| `order56/code/*/bench/`, `order78/code/bench/` | C++ timing benchmarks |
| `paper1/code/` | scripts that turn both pipelines into the tables, figures and quoted numbers of the paper |

The coefficients of all tableaux with 45 digits are in `order56/code/base/data/tableaux.json`,
`order56/code/base/data/tableaux_opt.json`, `order56/code/r1/data/tableaux_r1.json` and
`order78/data/tableaux.json`.

## Method names

The code was written before the names of the paper were fixed. `paper1/code/names.py` maps them:

| Code | Paper |
|---|---|
| `CRK5-opt`, `CRK5(4)` | CRK5-opt(4) |
| `CRK6-opt`, `CRK6(4)` | CRK6-opt(4) |
| `CRK7-opt`, `CRK7(5)` / `CRK7-s` | CRK7-opt(5) / CRK7-s(5) |
| `CRK8-opt`, `CRK8(6)` / `CRK8-s` | CRK8-opt(6) / CRK8-s(6) |
| `RK6-opt` | RK6-opt(4) |
| `Verner6(5)`, `Verner7(6)`, `Verner8(7)` | Verner78-6(5), Verner78-7(6), Verner78-8(7) (Verner 1978) |
| `Vern6(5)`, `Vern7`, `Vern8` | Vern6(5), Vern7(6), Vern8(7) (Verner 2010) |
| `DP5`, `CK5` | DP5(4), CK5(4) |

## Reproducing the tables and figures

Python 3.12 with the packages of `requirements.txt`:

```
pip install -r requirements.txt
cd paper1/code
python tables56.py
python tables78.py
python figs56.py
python figs78.py
python facts56.py
python facts78.py
```

The tables go to `paper1/paper/tables/`, the figures to `paper1/paper/figures/`, and the numbers quoted
in the text to `paper1/data/`. These scripts read the stored results of both pipelines and take a few
minutes.

## Recomputing the results

| Topic | Files |
|---|---|
| refinement of the tableaux; residuals, error constants, ranks, stability polynomials | `order56/code/base/code/check_tableaux.py`, `order78/code/check_tableaux.py` |
| Butcher's barrier over C (computer-algebra check) | `order56/code/base/code/barrier_proof_check.py` |
| singular-value spectra and tangent dimensions | `order56/code/r1/code/spectra_r1.py`, `order56/code/r1/code/structure_r1.py`, `order78/code/rank_mp.py` |
| projected D(1) and the Krawczyk test (orders five and six) | `order56/code/r1/code/d1_r1.py`, `order56/code/r1/code/krawczyk_r1.py`, `order56/code/r1/code/krawczyk_arb_r1.py`, `order56/code/r1/code/kernel_proof_r1.py` |
| confluent limit and the scheme L in 90 digits | `order78/code/confluent.py`, `order78/code/confluent_mp.py`, `order78/data/confluent_L.json` |
| minimization of the error constants and fronts | `order56/code/base/code/optimize_E.py`, `order56/code/base/code/front_E.py`, `order78/code/optimize_E.py`, `order78/code/front_E.py` |
| embedded formulas and pairs | `order56/code/base/code/embedded.py`, `order56/code/r1/code/real64_r1.py`, `order78/code/embedded.py` |
| stability and preference regions | `order56/code/base/code/stability.py`, `order78/code/stability.py`, `prefregions.py` in both |
| fixed-step error constants in 34 and 45 digits | `order56/code/base/code/measure.py`, `order56/code/r1/code/measure_r1.py`, `order78/code/measure.py` |
| variable-step runs and robustness checks | `order56/code/base/code/adaptive.py`, `order56/code/r1/code/adaptive_new_r1.py`, `order56/code/r1/code/robust_r1.py`, `order56/code/r1/code/traj_r1.py`, `order78/code/adaptive.py` |
| larger systems | `order56/code/r1/code/bruss2d_r1.py`, `order56/code/r1/code/nbody_r1.py`, `order78/code/lorenz96.py` |
| latency model, extrapolation and two-chain methods | `order56/code/r1/code/latency_r1.py`, `order56/code/r1/code/gbs_r1.py`, `order56/code/r1/code/gbs246_r1.py`, `order56/code/r1/code/two_chain_r1.py` |
| branch cuts on the Kepler and Arenstorf problems | `order56/code/r1/code/singular_r1.py`, `order78/code/branchcut.py` |
| C++ timing benchmark | `order56/code/r1/code/bench/`, `order78/code/bench/` |

Run each script from its own folder. The multiple-precision computations (90-digit spectra, Krawczyk
tests, 45-digit fixed-step runs) take from minutes to hours on a laptop; the CPU benchmark needs MSVC
and an otherwise idle machine. Rerunning a script overwrites its stored results.

## License

MIT, see `LICENSE`.
