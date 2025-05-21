import pickle
import argparse
import networkx as nx
import numpy as np
from collections import deque, defaultdict
import pandas as pd

from sais.sais_model import SAISModel


# ---------------- Gozinto Model ------------------
def compute_gozinto_influence(graph, affected_nodes, max_depth=10):
    influence = defaultdict(float)
    levels = defaultdict(lambda: defaultdict(set))
    for source in affected_nodes:
        visited = set()
        queue = deque([(source, 1)])
        while queue:
            node, depth = queue.popleft()
            if depth > max_depth or node in visited:
                continue
            visited.add(node)
            levels[source][depth].add(node)
            influence[node] += 1.0 / depth
            for neighbor in graph.successors(node):
                queue.append((neighbor, depth + 1))
    return dict(influence), dict(levels)


# ---------------- Affectation Transmission Model ------------------
def compute_affectation_transmission(graph, affected_nodes, threshold=0.5, decay=0.8):
    transmission = defaultdict(float)
    levels = defaultdict(lambda: defaultdict(set))
    queue = deque([(node, 1.0, 0, node) for node in affected_nodes])
    visited = set()

    while queue:
        current, value, level, origin = queue.popleft()
        if value < threshold:
            continue
        if (current, origin) in visited:
            continue
        visited.add((current, origin))
        transmission[current] += value
        levels[origin][level].add(current)
        for neighbor in graph.successors(current):
            queue.append((neighbor, value * decay, level + 1, origin))

    return dict(transmission), dict(levels)


# ---------------- SAIS Model ------------------
def run_sais_model(graph, affected_nodes, beta=0.1, gamma=0.05, delta=0.1, steps=15):
    sais = SAISModel(graph, beta=beta, gamma=gamma, delta=delta)
    history, state_history = sais.simulate_epidemic(steps=steps, infected_nodes=affected_nodes)

    # Build full DataFrame of states
    df_full = pd.DataFrame(index=range(steps), columns=list(graph.nodes))
    for t in range(steps):
        for node in graph.nodes:
            df_full.at[t, node] = state_history[t].get(node, 'S')
    df_full.index.name = "Step"

    # Filter only affected nodes (entered A or I at some point)
    affected_cols = [node for node in df_full.columns if any(df_full[node].isin(['A', 'I']))]
    df_filtered = df_full[affected_cols]

    # First step when each node was affected (A or I)
    node_first_step = {}
    for node in affected_cols:
        for t in range(steps):
            if df_filtered.at[t, node] in {'A', 'I'}:
                node_first_step[node] = t
                break

    # Breakdown by initial node
    levels = defaultdict(lambda: defaultdict(set))
    for node, step in node_first_step.items():
        for source in affected_nodes:
            if df_filtered.at[0, node] == 'S':  # exclude directly affected
                levels[source][step].add(node)

    df_levels = levels_to_dataframe(levels)

    # Final state: compute infected reachability
    final_infected = [node for node in df_filtered.columns if df_filtered.at[steps - 1, node] == 'I']
    infected_levels = defaultdict(lambda: defaultdict(set))
    for source in affected_nodes:
        visited = set()
        queue = deque([(source, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node in final_infected:
                infected_levels[source][depth].add(node)
            for neighbor in graph.successors(node):
                queue.append((neighbor, depth + 1))

    df_infected_levels = levels_to_dataframe(infected_levels)

    # Summary print
    step_summary = defaultdict(set)
    for node, step in node_first_step.items():
        step_summary[step].add(node)

    print("\nSAIS - Affected nodes per step:")
    for step in sorted(step_summary):
        print(f"  Step {step}: {len(step_summary[step])} nodes")

    return df_filtered, df_levels, df_infected_levels


# ---------------- Utility Functions ------------------
def print_level_summary(title, levels):
    print(f"\n{title} - Affected nodes per level:")
    summary = defaultdict(set)
    for level_data in levels.values():
        for level, nodes in level_data.items():
            summary[level].update(nodes)
    for level in sorted(summary):
        count = len(summary[level])
        print(f"  Level {level}: {count} nodes")


def levels_to_dataframe(levels_dict):
    all_levels = set()
    for source_levels in levels_dict.values():
        all_levels.update(source_levels.keys())
    all_levels = sorted(all_levels)

    df = pd.DataFrame(index=all_levels)
    for source, level_map in levels_dict.items():
        df[source] = [len(level_map.get(lvl, set())) for lvl in all_levels]
    df.index.name = "Depth"
    return df


# ---------------- Main CLI ------------------
def main():
    parser = argparse.ArgumentParser(description="Run Gozinto, Affectation Transmission and SAIS models on an OpenAlex citation graph.")
    parser.add_argument("graph_path", type=str, help="Path to the pickle file containing the citation graph.")
    parser.add_argument("--affected", nargs='+', required=True, help="List of affected OpenAlex IDs (e.g., W123...)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the pickle results.")
    parser.add_argument("--csv_prefix", type=str, default="model_output", help="Prefix for saving CSV output files.")
    parser.add_argument("--steps", type=int, default=15, help="Number of steps for SAIS model.")
    parser.add_argument("--beta", type=float, default=0.1, help="Infection rate for SAIS model.")
    parser.add_argument("--gamma", type=float, default=0.05, help="Recovery rate for SAIS model.")
    parser.add_argument("--delta", type=float, default=0.1, help="Alert rate for SAIS model.")
    args = parser.parse_args()

    with open(args.graph_path, "rb") as f:
        G = pickle.load(f)

    print("\nRunning Gozinto influence model...")
    gozinto, gozinto_levels = compute_gozinto_influence(G, args.affected)
    print_level_summary("Gozinto", gozinto_levels)
    df_gozinto = levels_to_dataframe(gozinto_levels)
    print("\nGozinto table:")
    print(df_gozinto)
    df_gozinto.to_csv(f"{args.csv_prefix}_gozinto.csv")

    print("\nRunning Affectation Transmission model...")
    transmission, transmission_levels = compute_affectation_transmission(G, args.affected)
    print_level_summary("Affectation Transmission", transmission_levels)
    df_transmission = levels_to_dataframe(transmission_levels)
    print("\nAffectation Transmission table:")
    print(df_transmission)
    df_transmission.to_csv(f"{args.csv_prefix}_transmission.csv")

    print("\nRunning SAIS model...")
    df_sais_states, df_sais_levels, df_sais_infected = run_sais_model(G, args.affected, beta=args.beta, gamma=args.gamma, delta=args.delta, steps=args.steps)
    df_sais_states.to_csv(f"{args.csv_prefix}_sais.csv")
    df_sais_levels.to_csv(f"{args.csv_prefix}_sais_levels.csv")
    df_sais_infected.to_csv(f"{args.csv_prefix}_sais_infected_levels.csv")
    print("SAIS simulation completed.")

    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump({
                "gozinto": gozinto,
                "transmission": transmission,
                "gozinto_levels": gozinto_levels,
                "transmission_levels": transmission_levels,
                "gozinto_table": df_gozinto,
                "transmission_table": df_transmission,
                "sais_states": df_sais_states,
                "sais_levels": df_sais_levels,
                "sais_infected_levels": df_sais_infected
            }, f)
        print(f"\nPickle results saved to {args.output}")

    print(f"\nCSV tables saved with prefix '{args.csv_prefix}'")


if __name__ == "__main__":
    main()
