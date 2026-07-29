# Technology Diffusion

Research code for studying technology diffusion on networks and finding small
sets of initially active nodes. The repository includes deterministic
heuristics, the Goldberg–Liu mixed-integer formulation, an approximation
algorithm, exact-IP validation, and **COaDS (Component-Aware Direct Search)**.

## Repository layout

- `src/technology_diffusion/`: the importable Python package.
  - `helpers.py`: graph generation, diffusion, and shared utilities.
  - `heuristics.py`: deterministic seed-selection heuristics.
  - `coads.py`: Component-Aware Direct Search.
  - `approx.py`: the approximation algorithm.
  - `ip_problems.py`: Goldberg–Liu and exact-IP formulations.
- `scripts/`: five maintained experiment runners.
  - `technology_diffusion.py`: main experiment.
  - `technology_diffusion_exact.py`: exact-IP validation on small graphs.
  - `technology_diffusion_heuristics.py`: deterministic-heuristic experiments.
  - `technology_diffusion_approx.py`: approximation experiments.
  - `technology_diffusion_benchmark.py`: diffusion-runtime benchmarks.
- `notebooks/`: maintained analysis and validation notebooks.
  - `technology_diffusion_results.ipynb`: result summaries, plots, and
    performance profiles.
  - `technology_diffusion_exact_ip_validation.ipynb`: exact-IP checks.
  - `technology_diffusion_approximation_validation.ipynb`: approximation
    checks.
- `results/` and `figures/`: generated outputs; both are ignored by Git.
- `misc/`: local archive for superseded scripts, notebooks, and other files.
  It is also ignored by Git.

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
python scripts/technology_diffusion_exact.py --help
python scripts/technology_diffusion_heuristics.py --help
python scripts/technology_diffusion_approx.py --help
python scripts/technology_diffusion_benchmark.py --help
```

The exact runner compares COaDS and Goldberg–Liu with the exact-IP solution on
small instances. The heuristic and approximation runners isolate their
respective algorithm families, while the benchmark measures diffusion-runtime
behavior.

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
