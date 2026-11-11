"""Headless AI-vs-AI evaluation (no pygame needed).

Examples:
    python tournament.py                       # Minimax(d=2) vs Greedy, 6 games
    python tournament.py --games 10 --depth 3 --time 2 --opponent random
    python tournament.py --opponent minimax --opp-depth 1   # depth 3 vs depth 1
"""
import argparse
import time

from quoridor import State, MinimaxAgent, GreedyAgent, RandomAgent


def make(kind, depth, tl):
    if kind == "minimax":
        return MinimaxAgent(depth, tl)
    return {"greedy": GreedyAgent, "random": RandomAgent}[kind]()


def play(agents, max_plies=300):
    """agents[p] controls player p. Returns (winner or None, plies, walls used)."""
    state = State()
    for ply in range(max_plies):
        w = state.winner()
        if w is not None:
            return w, ply
        state = state.apply(agents[state.turn].choose_move(state))
    return None, max_plies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--time", type=float, default=1.0, help="seconds per move")
    ap.add_argument("--opponent", choices=["greedy", "random", "minimax"], default="greedy")
    ap.add_argument("--opp-depth", type=int, default=1)
    args = ap.parse_args()

    wins = {"minimax": 0, "opponent": 0, "draw": 0}
    total_plies, t0 = 0, time.time()
    for g in range(args.games):
        mm = MinimaxAgent(args.depth, args.time)
        opp = make(args.opponent, args.opp_depth, args.time)
        mm_player = g % 2                      # alternate colours / first move
        agents = {mm_player: mm, 1 - mm_player: opp}
        winner, plies = play(agents)
        total_plies += plies
        key = "draw" if winner is None else ("minimax" if winner == mm_player else "opponent")
        wins[key] += 1
        print(f"game {g + 1}: minimax played {'first' if mm_player == 0 else 'second'}"
              f" -> {key} in {plies} plies")
    print(f"\nMinimax(d={args.depth}) vs {args.opponent}: {wins}  "
          f"avg plies {total_plies / args.games:.1f}  total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()