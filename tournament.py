"""Headless round-robin tournament (no pygame needed).

Agent specs:  random | greedy | minimax-basic:<depth> | minimax:<depth> | mcts:<iterations>

Examples:
    python tournament.py                                   # default line-up, 8 games per pair
    python tournament.py --agents greedy minimax-basic:2 minimax:2 mcts:400 --games 10
    python tournament.py --agents minimax:3 mcts:1000 --games 20 --openings 4

Fairness: every pair plays `--games` games.  Games come in colour-swapped
pairs that start from the same random opening (`--openings` random pawn plies),
so neither colour nor opening luck decides the result.  Strength is set by
search depth / MCTS iterations (not wall-clock time) so results are
reproducible; average time per move is reported separately.  A game that is
not finished after `--max-plies` is a draw.
"""
import argparse
import itertools
import random
import time

from quoridor import State, MinimaxAgent, MCTSAgent, GreedyAgent, RandomAgent

DEFAULT_AGENTS = ["random", "greedy", "minimax-basic:2", "minimax:2", "mcts:400"]


def make_agent(spec, seed):
    kind, _, arg = spec.partition(":")
    if kind == "random":
        return RandomAgent(seed)
    if kind == "greedy":
        return GreedyAgent()
    if kind == "minimax-basic":
        return MinimaxAgent(int(arg or 2), 60.0, improved=False)
    if kind == "minimax":
        return MinimaxAgent(int(arg or 2), 60.0, improved=True, seed=seed)
    if kind == "mcts":
        return MCTSAgent(int(arg or 400), seed=seed)
    raise SystemExit(f"unknown agent spec: {spec}")


def label(spec):
    return make_agent(spec, 0).name


def opening(seed, plies):
    """Random pawn-only opening, identical for both colour assignments."""
    rng = random.Random(seed)
    s = State()
    for _ in range(plies):
        s = s.apply(("move",) + rng.choice(s.pawn_moves()))
    return s


def play(agents, start, max_plies):
    """agents[p] controls player p.  Returns winner, plies, times[p], moves[p], walls[p]."""
    s = start
    times, moves, walls = [0.0, 0.0], [0, 0], [0, 0]
    for ply in range(max_plies):
        if s.winner() is not None:
            return s.winner(), ply, times, moves, walls
        p = s.turn
        t0 = time.perf_counter()
        mv = agents[p].choose_move(s)
        times[p] += time.perf_counter() - t0
        moves[p] += 1
        walls[p] += mv[0] == "wall"
        s = s.apply(mv)
    return s.winner(), max_plies, times, moves, walls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    ap.add_argument("--games", type=int, default=8, help="games per pair (use an even number)")
    ap.add_argument("--openings", type=int, default=2, help="random opening plies")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tournament_results.md")
    args = ap.parse_args()

    names = {spec: label(spec) for spec in args.agents}
    stat = {s: dict(w=0, d=0, l=0, t=0.0, m=0, walls=0, games=0) for s in args.agents}
    h2h = {(a, b): 0.0 for a in args.agents for b in args.agents if a != b}
    t_start = time.time()

    for a, b in itertools.combinations(args.agents, 2):
        res = {a: 0, b: 0, "draw": 0}
        plies_total = 0
        for g in range(args.games):
            seed = args.seed * 1000 + g // 2
            start = opening(seed, args.openings)
            a_player = g % 2
            ag = {a_player: make_agent(a, seed + 1), 1 - a_player: make_agent(b, seed + 2)}
            winner, plies, times, moves, walls = play(ag, start, args.max_plies)
            plies_total += plies
            for spec, pl in ((a, a_player), (b, 1 - a_player)):
                st = stat[spec]
                st["t"] += times[pl]; st["m"] += moves[pl]; st["walls"] += walls[pl]; st["games"] += 1
            if winner is None:
                res["draw"] += 1
                stat[a]["d"] += 1; stat[b]["d"] += 1
                h2h[(a, b)] += 0.5; h2h[(b, a)] += 0.5
            else:
                win, lose = (a, b) if winner == a_player else (b, a)
                res[win] += 1
                stat[win]["w"] += 1; stat[lose]["l"] += 1
                h2h[(win, lose)] += 1
        print(f"{names[a]:>22} vs {names[b]:<22} {res[a]:>2}-{res['draw']}-{res[b]:<2}"
              f"  avg plies {plies_total / args.games:5.1f}   [{time.time() - t_start:6.0f}s]", flush=True)

    rows = sorted(args.agents, key=lambda s: -(stat[s]["w"] + 0.5 * stat[s]["d"]))
    out = ["## Standings", "",
           "| Agent | Games | W | D | L | Score | Win % | ms/move | walls/game |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        st = stat[s]
        score = st["w"] + 0.5 * st["d"]
        out.append(f"| {names[s]} | {st['games']} | {st['w']} | {st['d']} | {st['l']} | {score:g} | "
                   f"{100 * score / st['games']:.0f}% | {1000 * st['t'] / max(1, st['m']):.1f} | "
                   f"{st['walls'] / st['games']:.1f} |")
    out += ["", "## Head-to-head (row's score against column, out of %d games)" % args.games, "",
            "| | " + " | ".join(names[s] for s in rows) + " |",
            "|---|" + "---|" * len(rows)]
    for a in rows:
        out.append(f"| **{names[a]}** | " + " | ".join(
            "-" if a == b else f"{h2h[(a, b)]:g}" for b in rows) + " |")
    out += ["", f"_{args.games} games per pair, {args.openings} random opening plies, "
                f"max {args.max_plies} plies, total {time.time() - t_start:.0f}s._"]
    text = "\n".join(out)
    print("\n" + text)
    with open(args.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
