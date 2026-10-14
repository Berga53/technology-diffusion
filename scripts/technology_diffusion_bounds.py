import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from gurobipy import GRB

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

RESULTS = ROOT / "results" / "technology diffusion"
RESULTS.mkdir(parents=True, exist_ok=True)

from technology_diffusion import (
	approx,
	build_golberg_liu_ip,
	connected_component_spread,
	create_pa_graph,
	get_activation_sequence_and_seeds,
	make_vector,
)

DEFAULT_C_LIST = [1, 2, 3, 4]
DEFAULT_N_LIST = [6, 8, 10, 12, 15, 20]
DEFAULT_SEED_LIST = list(range(10))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compare brute-force OPT, Goldberg–Liu (connected activation sequences), "
			"the approximation algorithm and the bound min{2OPT, OPT+ell} on small graphs."
		)
	)
	parser.add_argument("--c-list", type=int, nargs="+", default=DEFAULT_C_LIST)
	parser.add_argument("--n-list", type=int, nargs="+", default=DEFAULT_N_LIST)
	parser.add_argument("--seed-list", type=int, nargs="+", default=DEFAULT_SEED_LIST)
	parser.add_argument("--init-nodes", type=int, default=5)
	parser.add_argument("--init-mode", choices=["complete", "tree"], default="complete")
	parser.add_argument(
		"--results-csv-path",
		type=Path,
		default=RESULTS / "approx_goldberg_liu_vs_bruteforce.csv",
	)
	parser.add_argument(
		"--static-params-path",
		type=Path,
		default=RESULTS / "approx_goldberg_liu_vs_bruteforce_static_params.json",
	)
	parser.add_argument("--save-results-csv", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def activates_all(g: nx.Graph, thetas: dict[int, int], seeds) -> bool:
	n_nodes = g.number_of_nodes()
	x0 = make_vector(seeds, n_nodes)
	spread, _, _ = connected_component_spread(g, x0, thetas)
	return spread == n_nodes


def brute_force_optimum(
	g: nx.Graph,
	thetas: dict[int, int],
) -> tuple[int, tuple[int, ...], int, float]:
	"""Return unrestricted (optimal size, one optimal set, subsets tested, runtime)."""
	start = time.perf_counter()
	nodes = tuple(sorted(g.nodes()))
	if nodes != tuple(range(len(nodes))):
		raise ValueError("The diffusion evaluator expects nodes labelled 0, ..., n-1.")

	evaluated = 0
	for k in range(1, len(nodes) + 1):
		for seed_tuple in itertools.combinations(nodes, k):
			evaluated += 1
			if activates_all(g, thetas, seed_tuple):
				return k, seed_tuple, evaluated, time.perf_counter() - start

	raise RuntimeError("No feasible seed set was found.")


def solve_goldberg_liu(
	g: nx.Graph,
	thetas: dict[int, int],
) -> tuple[int, tuple[int, ...], tuple[int, ...], float]:
	"""Build and solve the Goldberg–Liu IP; return (objective, seeds, order, runtime)."""
	start = time.perf_counter()
	model, xvars = build_golberg_liu_ip(g, thetas)
	if model is None or xvars is None:
		raise RuntimeError("The Goldberg–Liu model could not be built.")

	model.Params.OutputFlag = 0
	model.optimize()
	if model.Status != GRB.OPTIMAL:
		status = model.Status
		model.dispose()
		raise RuntimeError(f"Goldberg–Liu did not solve to optimality (status={status}).")

	_, order, seeds = get_activation_sequence_and_seeds(g, thetas, xvars)
	objective = int(round(model.ObjVal))
	runtime = time.perf_counter() - start
	model.dispose()

	seeds = tuple(sorted(int(u) for u in seeds))
	order = tuple(int(u) for u in order)
	if objective != len(seeds):
		raise AssertionError("Goldberg–Liu objective and extracted seed set disagree.")
	if not activates_all(g, thetas, seeds):
		raise AssertionError("The Goldberg–Liu seed set is not feasible under the evaluator.")
	return objective, seeds, order, runtime


def run_instance(n_nodes: int, c: int, seed: int, init_nodes: int, init_mode: str) -> dict:
	g, thetas = create_pa_graph(
		n_nodes=n_nodes,
		c=c,
		seed=seed,
		init_nodes=init_nodes,
		init_mode=init_mode,
	)
	threshold_values = tuple(sorted(set(int(v) for v in thetas.values())))
	ell = len(threshold_values)

	opt_k, opt_seeds, opt_tests, opt_runtime = brute_force_optimum(g, thetas)
	gl_k, gl_seeds, gl_order, gl_runtime = solve_goldberg_liu(g, thetas)

	approx_k, approx_x, approx_runtime, _ = approx(g, thetas)
	approx_success = approx_k is not None and approx_x is not None
	approx_seeds = tuple(int(u) for u in np.flatnonzero(approx_x)) if approx_success else tuple()
	approx_feasible = (
		activates_all(g, thetas, approx_seeds)
		and (len(approx_seeds) <= 1 or nx.is_connected(g.subgraph(approx_seeds)))
		if approx_success
		else False
	)

	bound_2opt = 2 * opt_k
	bound_opt_plus_ell = opt_k + ell
	theory_bound = min(bound_2opt, bound_opt_plus_ell)

	return {
		"n_nodes": n_nodes,
		"c": c,
		"seed": seed,
		"n_edges": g.number_of_edges(),
		"threshold_values": threshold_values,
		"ell": ell,
		"opt_bruteforce": opt_k,
		"approx_k": int(approx_k) if approx_success else np.nan,
		"goldberg_liu_k": gl_k,
		"bound_2opt": bound_2opt,
		"bound_opt_plus_ell": bound_opt_plus_ell,
		"theory_bound": theory_bound,
		"active_bound_branch": "2OPT" if bound_2opt <= bound_opt_plus_ell else "OPT+ell",
		"opt_seeds": opt_seeds,
		"approx_seeds": approx_seeds,
		"goldberg_liu_seeds": gl_seeds,
		"goldberg_liu_order": gl_order,
		"opt_bf_subsets_tested": opt_tests,
		"opt_bf_runtime_s": opt_runtime,
		"approx_runtime_s": approx_runtime,
		"goldberg_liu_runtime_s": gl_runtime,
		"approx_success": approx_success,
		"approx_feasible": approx_feasible,
		"goldberg_is_upper_bound": gl_k >= opt_k,
		"bound_holds_for_goldberg": gl_k <= theory_bound,
	}


def main() -> None:
	args = parse_args()
	combinations = list(itertools.product(args.n_list, args.c_list, args.seed_list))

	print(f"Total graph runs: {len(combinations)}")
	print(f"Results CSV: {args.results_csv_path}")

	rows = []
	start_all = time.perf_counter()
	for run_idx, (n_nodes, c, seed) in enumerate(combinations, start=1):
		row = run_instance(n_nodes, c, seed, args.init_nodes, args.init_mode)
		rows.append(row)
		approx_txt = "fail" if np.isnan(row["approx_k"]) else int(row["approx_k"])
		print(
			f"[{run_idx}/{len(combinations)}] N={n_nodes}, c={c}, seed={seed} | "
			f"OPT={row['opt_bruteforce']} GL={row['goldberg_liu_k']} approx={approx_txt} "
			f"bound={row['theory_bound']}"
		)

	df = pd.DataFrame(rows)
	df["approx_k_for_plot"] = df["approx_k"].fillna(df["n_nodes"])
	df["goldberg_to_opt_ratio"] = df["goldberg_liu_k"] / df["opt_bruteforce"]
	df["approx_to_opt_ratio"] = df["approx_k_for_plot"] / df["opt_bruteforce"]
	df["bound_to_opt_ratio"] = df["theory_bound"] / df["opt_bruteforce"]
	df["bound_slack_for_goldberg"] = df["theory_bound"] - df["goldberg_liu_k"]

	elapsed_all = time.perf_counter() - start_all
	print("\n" + "=" * 90)
	print(f"Completed {len(df)} instances in {round(elapsed_all, 2)} seconds.")
	print(f"Goldberg–Liu >= OPT:          {df['goldberg_is_upper_bound'].mean():.1%}")
	print(f"Goldberg–Liu <= bound:        {df['bound_holds_for_goldberg'].mean():.1%}")
	print(f"Approximation success rate:   {df['approx_success'].mean():.1%}")

	if args.save_results_csv:
		df.to_csv(args.results_csv_path, index=False)
		print(f"Results saved to: {args.results_csv_path}")

	static_params = {
		"n_list": args.n_list,
		"c_list": args.c_list,
		"seed_list": args.seed_list,
		"init_nodes": args.init_nodes,
		"init_mode": args.init_mode,
		"result_columns": list(df.columns),
	}
	with args.static_params_path.open("w", encoding="utf-8") as file:
		json.dump(static_params, file, indent=2)
	print(f"Static parameters saved to: {args.static_params_path}")


if __name__ == "__main__":
	main()
