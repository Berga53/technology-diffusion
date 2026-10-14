# Technology Diffusion

Research code for studying technology diffusion on networks and finding small
sets of initially active nodes. The repository includes deterministic
heuristics, the Goldberg–Liu mixed-integer formulation, an approximation
algorithm, a connectivity-bound experiment on small graphs, and
**COaDS (Component-Aware Direct Search)**.

## Repository layout

- `src/technology_diffusion/`: the importable Python package.
  - `helpers.py`: graph generation, diffusion, and shared utilities.
  - `heuristics.py`: deterministic seed-selection heuristics.
  - `coads.py`: Component-Aware Direct Search.
  - `approx.py`: the approximation algorithm.
  - `ip_problems.py`: Goldberg–Liu and exact-IP formulations.
- `technology_diffusion_results.ipynb`: the main results notebook, with
  summaries, plots, statistics, relative-gap analysis, and performance
  profiles.
- `scripts/`: three maintained experiment runners.
  - `technology_diffusion.py`: main experiment.
  - `technology_diffusion_benchmark.py`: diffusion-runtime benchmarks.
  - `technology_diffusion_bounds.py`: connectivity bound on small graphs
    (brute-force OPT vs. Goldberg–Liu vs. min{2OPT, OPT+ℓ}).
  
## Setup

The code has been tested with Python 3.13.7. Create the project environment
from the repository root:

```bash
make setup PYTHON=python3.13
```

Alternatively:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The IP-based algorithms require a working Gurobi installation and license.
For the notebooks, also install Jupyter and the plotting dependencies:

```bash
python -m pip install jupyter matplotlib seaborn
```

## Run the main experiment

`scripts/technology_diffusion.py` is the main entry point:

```bash
source .venv/bin/activate
python scripts/technology_diffusion.py
```

The equivalent Make target runs it through the local virtual environment:

```bash
make run-technology-diffusion
```

View all command-line options with:

```bash
make run-technology-diffusion-help
```

For a smaller custom run, pass explicit graph sizes, values of `c`, and seeds:

```bash
python scripts/technology_diffusion.py \
  --n-list 200 \
  --c-list 5 \
  --seed-list 42
```

The main experiment evaluates six deterministic heuristics, Goldberg–Liu, the
approximation algorithm, and COaDS. Its default seed list is taken from the
existing result files:

```text
1, 2, 3, 5, 7, 11, 13, 15, 16, 17, 19, 25, 32, 33,
42, 53, 64, 71, 75, 98, 99, 101, 104, 111, 118
```

The default graph sizes are `200, 400, 600, 1000, 2000`, and the default
values of `c` are `1, 5, 10, 20`. These defaults can produce a long run, so use
the CLI filters when testing changes.

## Other experiment scripts

Each specialized runner has its own `--help` output:

```bash
python scripts/technology_diffusion_benchmark.py --help
python scripts/technology_diffusion_bounds.py --help
```

The benchmark runner measures diffusion-runtime behavior.

The bounds runner compares, on small graphs (by default `n = 6, 8, 10, 12, 15, 20`,
`c = 1, 2, 3, 4`, seeds `0-9`), the unrestricted optimum found by brute force,
the Goldberg–Liu IP solution (connected activation sequences, not guaranteed
to reach the connected optimum), and the bound
`min{2OPT, OPT+ℓ}`. It writes `approx_goldberg_liu_vs_bruteforce.csv`, which the
"Connectivity Bound" section of the results notebook plots.

## Results

By default, experiment files are written below:

```text
results/technology diffusion/
```

The main long-form CSV uses the columns:

```text
algorithm, n_nodes, c, seed, total_time, K, history
```

COaDS results use the algorithm label `coads_binary_search`. Output filenames
are tagged automatically with the experiment parameters to reduce accidental
overwrites. Use `--no-auto-name-outputs` when a fixed filename is preferred;
the runners also provide options for overriding individual output paths.

Open the root-level results notebook after generating the experiment data:

```bash
jupyter notebook technology_diffusion_results.ipynb
```

## Importing the package

The Python package uses the standard importable name `technology_diffusion`.
The maintained scripts configure the `src` path automatically. For an
interactive session from the repository root, use:

```python
import sys

sys.path.insert(0, "src")
from technology_diffusion import approx, create_pa_graph
```

The project and generated results retain the human-readable name
“Technology Diffusion.” Quote the results-directory path when using it in a
shell because it contains a space.
