"""Performance analysis of the Quoridor algorithms (beyond win / loss counts).

Experiments (run all, or pick with --exp):

  astar     A* vs uninformed search (Dijkstra / BFS): nodes expanded and time
            as the number of walls on the board grows.
  branching How big is the move set?  all legal walls -> walls on the opponent's
            path -> walls that really help -> walls actually searched.
  depth     Minimax cost vs search depth: nodes, time, effective branching
            factor, transposition-table size, early/middle/late game.
  ablation  Which optimisation buys what?  (alpha-beta, move ordering,
            transposition table, wall pruning, improved evaluation) - also checks
            that alpha-beta returns exactly the same value as plain minimax.
  mcts      MCTS cost vs number of iterations (time, tree size, tree depth).
  quality   Decision quality without playing games: every agent picks a move on
            the same positions and is scored by "regret" against a depth-4
            reference search.  (Biased towards Minimax, see the note in the output.)
  strength  Game-play strength: round-robin tournament between every algorithm
            (Random, Greedy A*, Minimax basic/improved, MCTS) with win rates,
            95% confidence intervals, p-values, time per move, walls per game.
  stats     Confidence intervals / p-values for results you already recorded.

Usage:
    python analysis.py                      # everything (the tournament dominates the time)
    python analysis.py --quick              # fewer positions, shallower, tiny tournament
    python analysis.py --exp strength --games 20 --agents greedy minimax:3 mcts:1000
    python analysis.py --exp astar,depth,ablation,mcts,quality   # everything except games
    python analysis.py --exp astar,depth    # selected experiments
    python analysis.py --match 25 15        # significance of "25 wins, 15 losses"

Output: analysis_output/analysis_results.md, CSV files and PNG charts
(charts need matplotlib; everything else needs only the standard library).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import time
from collections import deque
from heapq import heappop, heappush

from quoridor.agent import (WIN_SCORE, GreedyAgent, MinimaxAgent, RandomAgent,
                            candidate_walls)
from quoridor import game as game_module
from quoridor.game import State
from quoridor.mcts import MCTSAgent

BIG = 1e9          # "no time limit": we want to measure the work, not the clock
CLIP = 100         # regret study: values are clipped to +-100 (10 squares) so one lost game isn't worth 10 000


# =============================================================== utilities
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fmt(x, nd=1):
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def binom_p(w, n):
    """Exact two-sided p-value for H0: both players equally strong (p = 0.5)."""
    k = max(w, n - w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def elo_diff(p):
    return 400 * math.log10(p / (1 - p)) if 0 < p < 1 else float("inf") * (1 if p >= 1 else -1)


def result_row(label, w, n):
    lo, hi = wilson(w, n)
    return [label, f"{w}/{n}", f"{100 * w / n:.0f}%", f"{100 * lo:.0f}-{100 * hi:.0f}%",
            f"{binom_p(w, n):.3f}", f"{elo_diff(w / n):+.0f}"]


# results already collected in the project's tournaments (wins of the FIRST agent / games)
KNOWN_RESULTS = [
    ("Search agents vs Greedy A* (round-robin, pooled)", 39, 40),   # Greedy won only 1 of its 40 games
    ("MCTS(n=1500) beat Minimax(d=3)", 25, 40),
    ("MCTS(n=300) beat Minimax(d=5)", 23, 40),
    ("MCTS pooled over both 40-game matches", 48, 80),
]


# ============================================================ test positions
def realistic_board(k, rng):
    """k walls placed the way real games place them: across a player's shortest path."""
    s, placed, tries = State(), 0, 0
    while placed < k and tries < k * 40:
        tries += 1
        target = rng.randrange(2)
        cands = candidate_walls(s, target)
        if not cands:
            continue
        r, c, o = rng.choice(cands)
        nxt = s.try_wall(r, c, o, player=1 - target)
        if nxt is not None:
            s, placed = nxt, placed + 1
    return s


def sample_positions(n, seed=7):
    """n test positions spread from opening to late game: (phase, State).

    Early = few walls and pawns near their start, late = many walls and pawns
    advanced.  Self-play games from our agents are too short to give a fair
    spread, so positions are built directly (walls block real shortest paths).
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        frac = i / max(1, n - 1)
        label = "early" if frac < 1 / 3 else "middle" if frac < 2 / 3 else "late"
        k, adv = round(2 + 14 * frac), round(1 + 3 * frac)
        for _ in range(100):
            b = realistic_board(k, rng)
            p0 = (8 - adv - rng.randint(0, 1), rng.randint(2, 6))
            p1 = (adv + rng.randint(0, 1), rng.randint(2, 6))
            t = State((p0, p1), b.walls_left, b.walls, rng.randrange(2))
            if p0 != p1 and t.dist(0) is not None and t.dist(1) is not None and min(t.dist(0), t.dist(1)) >= 2:
                out.append((label, t))
                break
    return out


def timed(agent, state):
    game_module._PATH_CACHE.clear()      # cold start: don't let earlier runs pre-compute A* paths
    t = time.perf_counter()
    mv = agent.choose_move(state)
    return mv, time.perf_counter() - t


# =============================================================== A* study
def search_counted(state, start, goal_row, mode):
    """Returns (distance, nodes expanded).  mode: astar | astar_deep | dijkstra | bfs."""
    if mode == "bfs":
        q, seen, exp = deque([(start, 0)]), {start}, 0
        while q:
            cur, d = q.popleft()
            exp += 1
            if cur[0] == goal_row:
                return d, exp
            for nb in state.neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return None, exp
    h = (lambda c: abs(c[0] - goal_row)) if mode.startswith("astar") else (lambda c: 0)
    sign = -1 if mode == "astar_deep" else 1          # tie-break: prefer deeper nodes
    heap, best, closed, exp = [(h(start), 0, start)], {start: 0}, set(), 0
    while heap:
        _, g, cur = heappop(heap)
        g = abs(g)
        if cur in closed:
            continue
        exp += 1
        if cur[0] == goal_row:
            return g, exp
        closed.add(cur)
        for nb in state.neighbors(cur):
            ng = g + 1
            if ng < best.get(nb, 99):
                best[nb] = ng
                heappush(heap, (ng + h(nb), sign * ng, nb))
    return None, exp


def random_board(k, rng):
    s = State()
    placed = tries = 0
    while placed < k and tries < k * 12:
        tries += 1
        nxt = s.try_wall(rng.randrange(8), rng.randrange(8), rng.choice("HV"), player=rng.randrange(2))
        if nxt is not None:
            s, placed = nxt, placed + 1
    pawns = ((rng.randint(4, 8), rng.randrange(9)), (rng.randint(0, 4), rng.randrange(9)))
    return State(pawns, s.walls_left, s.walls, 0)


def exp_astar(rep, quick):
    rng = random.Random(1)
    modes = [("astar", "A* (project: ties -> shallow first)"), ("astar_deep", "A* (ties -> deep first)"),
             ("dijkstra", "Dijkstra (h = 0)"), ("bfs", "BFS")]
    boards_per_k = 60 if quick else 200
    rows, csv_rows = [], []
    chart = {m: [] for m, _ in modes}
    ks = [0, 5, 10, 15, 20]
    for k in ks:
        stats = {m: [0, 0.0] for m, _ in modes}
        count = 0
        for _ in range(boards_per_k):
            s = random_board(k, rng)
            for p, goal in ((0, 0), (1, 8)):
                dists = set()
                for m, _ in modes:
                    reps = 3
                    t = time.perf_counter()
                    for _ in range(reps):
                        d, exp = search_counted(s, s.pawns[p], goal, m)
                    dt = (time.perf_counter() - t) / reps
                    stats[m][0] += exp
                    stats[m][1] += dt
                    dists.add(d)
                assert len(dists) == 1, f"algorithms disagree on distance: {dists}"
                count += 1
        for m, label in modes:
            e, t = stats[m][0] / count, stats[m][1] / count * 1e6
            chart[m].append(e)
            rows.append([k, label, fmt(e), fmt(t)])
            csv_rows.append([k, m, e, t])
    write_csv(os.path.join(rep.out, "astar.csv"), ["walls_on_board", "algorithm", "nodes_expanded", "microseconds"], csv_rows)
    rep.section("1. Path-finding: A* against uninformed search",
                f"Shortest path from a random pawn position to its goal row on {boards_per_k * 2} random boards "
                "per wall count. All four algorithms return the same distance (checked on every instance); "
                "they differ in how many cells they expand.\n\n"
                + md_table(["Walls on board", "Algorithm", "Nodes expanded", "Time (µs)"], rows))
    rep.chart("astar.png", lambda plt: _plot_lines(plt, ks, chart, dict(modes), "Walls on board",
                                                   "Mean nodes expanded", "A* vs uninformed search"))
    base = chart["bfs"][-1]
    rep.note(f"With 20 walls A* expands {100 * chart['astar'][-1] / base:.0f}% of the cells BFS expands "
             f"(deep-first tie-breaking: {100 * chart['astar_deep'][-1] / base:.0f}%).")


# ============================================================= branching
def exp_branching(rep, positions):
    rows, csv_rows = [], []
    agg = {"pawn": [], "all": [], "path": [], "useful": [], "kept": []}
    for ply, s in positions:
        p, opp = s.turn, 1 - s.turn
        pawn = len(s.pawn_moves(p))
        allw = len(s.legal_walls()) if s.walls_left[p] else 0
        base = s.dist(opp)
        path = useful = 0
        for r, c, o in candidate_walls(s, opp):
            ch = s.try_wall(r, c, o)
            if ch is None:
                continue
            path += 1
            if ch.dist(opp) > base:
                useful += 1
        if not s.walls_left[p]:
            path = useful = 0
        kept = min(useful, 10)
        for k, v in zip(agg, (pawn, allw, path, useful, kept)):
            agg[k].append(v)
        csv_rows.append([ply, pawn, allw, path, useful, kept])
    write_csv(os.path.join(rep.out, "branching.csv"), ["phase", "pawn", "all_walls", "path_walls", "useful_walls", "kept"], csv_rows)
    m = {k: statistics.mean(v) for k, v in agg.items()}
    stages = [("Every legal move (pawn + all walls)", m["pawn"] + m["all"]),
              ("Pawn moves + walls crossing the opponent's path", m["pawn"] + m["path"]),
              ("... only walls that really lengthen the path", m["pawn"] + m["useful"]),
              ("... best 10 walls kept (improved agent)", m["pawn"] + m["kept"])]
    full = stages[0][1]
    rows = [[n, fmt(v), f"{100 * v / full:.0f}%"] for n, v in stages]
    rep.section("2. Branching factor and the effect of wall pruning",
                f"Mean number of moves per node on {len(positions)} sampled positions "
                "(early / middle / late game; walls placed across real shortest paths).\n\n" + md_table(["Move set", "Mean moves", "% of all"], rows))
    rep.chart("branching.png", lambda plt: _plot_bars(plt, [n for n, _ in stages], [v for _, v in stages],
                                                       "Moves per node", "Move generation funnel", rot=15))
    rep.note(f"Pruning shrinks the branching factor from about {full:.0f} to about {stages[-1][1]:.0f}; "
             "with a tree of b^d nodes this is the difference between depth 2 and depth 4+ being affordable.")


# ======================================================== minimax vs depth
def exp_depth(rep, positions, depths):
    per = {d: [] for d in depths}
    rows, csv_rows = [], []
    for d in depths:
        for ply, s in positions:
            ag = MinimaxAgent(d, BIG, seed=0)
            _, dt = timed(ag, s)
            per[d].append((ply, ag.last_stats["nodes"], dt, len(ag._tt)))
    prev = None
    for d in depths:
        nodes = statistics.mean(x[1] for x in per[d])
        tms = statistics.mean(x[2] for x in per[d])
        tt = statistics.mean(x[3] for x in per[d])
        ebf = f"{nodes / prev:.1f}x" if prev else "-"
        rows.append([d, fmt(nodes, 0), fmt(tms * 1000, 0), ebf, fmt(tt, 0)])
        csv_rows.append([d, nodes, tms, tt])
        prev = nodes
    write_csv(os.path.join(rep.out, "depth.csv"), ["depth", "mean_nodes", "mean_seconds", "mean_tt_entries"], csv_rows)
    body = ("Improved Minimax (alpha-beta + iterative deepening + TT), fresh agent per position, "
            f"no time limit, {len(positions)} positions. 'Growth' = nodes(d) / nodes(d-1); "
            "nodes include the shallower iterations of iterative deepening. A node also pays for generating and "
            "ordering its children (two A* paths each), so time grows a little faster than nodes.\n\n"
            + md_table(["Depth", "Mean nodes", "Mean time (ms)", "Growth", "TT entries"], rows))
    dmax = depths[-1]
    prow = []
    for ph in ("early", "middle", "late"):
        sel = [x for x in per[dmax] if x[0] == ph]
        if sel:
            prow.append([ph, len(sel), fmt(statistics.mean(x[1] for x in sel), 0),
                         fmt(statistics.mean(x[2] for x in sel) * 1000, 0)])
    body += f"\n\nBy game phase at depth {dmax}:\n\n" + md_table(["Phase", "Positions", "Mean nodes", "Mean time (ms)"], prow)
    rep.section("3. Minimax cost against search depth", body)
    rep.chart("depth.png", lambda plt: _plot_depth(plt, depths, per))
    g = [statistics.mean(x[1] for x in per[b]) / statistics.mean(x[1] for x in per[a]) for a, b in zip(depths, depths[1:])]
    rep.note(f"Each extra ply multiplies the work by about {statistics.mean(g):.1f}x (effective branching factor), "
             "which is why the time limit + iterative deepening matters in the GUI.")


# ================================================================ ablation
class PlainMinimax(MinimaxAgent):
    """Reference: fixed-depth minimax, no pruning / ordering / TT, basic evaluation."""

    def __init__(self, depth):
        super().__init__(depth, BIG, improved=False)
        self.name = f"Plain minimax(d={depth})"

    def _plain(self, state, depth, me):
        self._nodes += 1
        w = state.winner()
        if w is not None:
            return (WIN_SCORE + depth) if w == me else -(WIN_SCORE + depth)
        if depth == 0:
            return self.evaluate(state, me)
        vals = [self._plain(ch, depth - 1, me) for _, ch in self._children(state, me, order=False)]
        return max(vals) if state.turn == me else min(vals)

    def choose_move(self, state):
        t0, self._nodes, me = time.time(), 0, state.turn
        best, bv = None, float("-inf")
        for mv, ch in self._children(state, me, order=False):
            v = self._plain(ch, self.max_depth - 1, me)
            if v > bv:
                best, bv = mv, v
        self.last_stats = {"depth": self.max_depth, "nodes": self._nodes, "time": time.time() - t0, "score": bv}
        return best


class NoOrderMinimax(MinimaxAgent):
    """Alpha-beta without move ordering."""

    def _children(self, state, me, order, first=None):
        return super()._children(state, me, False, first)


def exp_ablation(rep, positions, depth):
    variants = [
        ("Full improved agent", lambda: MinimaxAgent(depth, BIG, seed=0)),
        ("- transposition table", lambda: MinimaxAgent(depth, BIG, seed=0, use_tt=False)),
        ("- move ordering", lambda: NoOrderMinimax(depth, BIG, seed=0)),
        ("- wall cap (search all useful walls)", lambda: MinimaxAgent(depth, BIG, seed=0, max_walls=99)),
        ("- 'useful wall' filter (any path wall)", lambda: MinimaxAgent(depth, BIG, seed=0, useful_only=False)),
        ("Basic evaluation (original agent)", lambda: MinimaxAgent(depth, BIG, seed=0, improved=False)),
        ("Plain minimax (no pruning, no ordering)", lambda: PlainMinimax(depth)),
    ]
    res = {}
    for name, make in variants:
        nodes, secs = [], []
        for ply, s in positions:
            ag = make()
            _, dt = timed(ag, s)
            nodes.append(ag.last_stats["nodes"])
            secs.append(dt)
        res[name] = (statistics.mean(nodes), statistics.mean(secs))
    base_n, base_t = res[variants[0][0]]
    rows = [[n, fmt(v[0], 0), f"{v[0] / base_n:.2f}x", fmt(v[1] * 1000, 0), f"{v[1] / base_t:.2f}x"] for n, v in res.items()]
    write_csv(os.path.join(rep.out, "ablation.csv"), ["variant", "mean_nodes", "mean_seconds"],
              [[n, v[0], v[1]] for n, v in res.items()])
    body = (f"Depth {depth}, {len(positions)} positions, no time limit. Ratios are relative to the full agent "
            "(above 1x = more work).\n\n" + md_table(["Variant", "Mean nodes", "vs full", "Mean time (ms)", "vs full"], rows))

    # correctness: alpha-beta (basic eval) must return exactly the plain-minimax value
    ok = tot = 0
    sp = []
    for d in range(1, min(depth, 3) + 1):
        for ply, s in positions:
            ab, pl = MinimaxAgent(d, BIG, improved=False), PlainMinimax(d)
            ab.choose_move(s)
            pl.choose_move(s)
            tot += 1
            ok += ab.last_stats["score"] == pl.last_stats["score"]
            if d == min(depth, 3):
                sp.append(pl.last_stats["nodes"] / max(1, ab.last_stats["nodes"]))
    body += (f"\n\n**Correctness check:** alpha-beta returned the same root value as plain minimax in "
             f"{ok}/{tot} (position, depth) cases. At depth {min(depth, 3)} alpha-beta with ordering visits "
             f"{statistics.mean(sp):.1f}x fewer nodes than plain minimax on the same move set "
             "(iterative-deepening overhead included).")
    rep.section("4. Ablation: what does each optimisation buy?", body)
    names = [n for n in res]
    rep.chart("ablation.png", lambda plt: _plot_bars(plt, names, [res[n][0] for n in names],
                                                     "Mean nodes searched", f"Ablation at depth {depth}", rot=25, log=True))


# =================================================================== MCTS
def exp_mcts(rep, positions, iters):
    rows, csv_rows = [], []
    for it in iters:
        tms, nodes, dep = [], [], []
        for ply, s in positions:
            ag = MCTSAgent(it, seed=0)
            _, dt = timed(ag, s)
            tms.append(dt)
            nodes.append(ag.last_stats.get("nodes", 0))
            dep.append(ag.last_stats.get("depth", 0))
        ms = statistics.mean(tms) * 1000
        rows.append([it, fmt(ms, 0), fmt(ms / it, 2), fmt(statistics.mean(nodes), 0), fmt(statistics.mean(dep), 1)])
        csv_rows.append([it, statistics.mean(tms), statistics.mean(nodes), statistics.mean(dep)])
    write_csv(os.path.join(rep.out, "mcts.csv"), ["iterations", "mean_seconds", "tree_nodes", "tree_depth"], csv_rows)
    rep.section("5. MCTS cost against iteration budget",
                f"{len(positions)} positions, no time limit.\n\n"
                + md_table(["Iterations", "Mean time (ms)", "ms / iteration", "Tree nodes", "Tree depth"], rows))
    rep.chart("mcts.png", lambda plt: _plot_xy(plt, iters, [float(r[1].replace(",", "")) for r in rows],
                                               "MCTS iterations", "Mean time per move (ms)", "MCTS cost"))
    rep.note("MCTS time grows linearly with the iteration budget and its memory with the tree size; "
             "Minimax cost grows exponentially with depth but is bounded by pruning and the time limit.")


# ================================================================ quality
def exp_quality(rep, positions, depth_ref, mcts_iters, minimax_depths):
    """Regret of each agent's move against a depth-`depth_ref` reference search."""
    agents = [("Random", lambda: RandomAgent(seed=1)), ("Greedy A*", lambda: GreedyAgent())]
    agents += [(f"Minimax(d={d})", (lambda d=d: MinimaxAgent(d, BIG, seed=0))) for d in minimax_depths]
    agents += [("Minimax(d=2, basic eval)", lambda: MinimaxAgent(2, BIG, seed=0, improved=False))]
    agents += [(f"MCTS(n={n})", (lambda n=n: MCTSAgent(n, seed=0))) for n in mcts_iters]
    regrets = {n: [] for n, _ in agents}
    for ply, s in positions:
        me = s.turn
        ref = MinimaxAgent(depth_ref, BIG, seed=0)
        ref._deadline = time.time() + BIG
        cache = {}

        def value(mv):
            if mv not in cache:
                v = ref._minimax(s.apply(mv, check=False), depth_ref - 1, float("-inf"), float("inf"), me)
                cache[mv] = max(-CLIP, min(CLIP, v))          # a forced win/loss counts as +-CLIP
            return cache[mv]

        for mv, _ in ref._children(s, me, order=True):
            value(mv)
        picks = {n: mk().choose_move(s) for n, mk in agents}
        for mv in picks.values():
            value(mv)
        best = max(cache.values())
        for n, mv in picks.items():
            regrets[n].append(best - cache[mv])
    rows, csv_rows = [], []
    for n, _ in agents:
        r = regrets[n]
        rows.append([n, fmt(statistics.mean(r), 1), fmt(statistics.median(r), 1),
                     f"{100 * sum(x < 1e-6 for x in r) / len(r):.0f}%",
                     f"{100 * sum(x >= 20 for x in r) / len(r):.0f}%"])
        csv_rows.append([n, statistics.mean(r), statistics.median(r)])
    write_csv(os.path.join(rep.out, "quality.csv"), ["agent", "mean_regret", "median_regret"], csv_rows)
    rep.section("6. Decision quality (regret against a deeper search)",
                f"On {len(positions)} positions every agent chooses a move; each move is scored by the value of the "
                f"resulting position under a depth-{depth_ref} reference search (evaluation units; 10 = one square of "
                "path length; forced wins/losses are clipped to +-100). Regret = best available value - value of the chosen "
                "move (lower is better). A blunder costs at least two squares of path length.\n\n"
                + md_table(["Agent", "Mean regret", "Median regret", "Best-move rate", "Blunders (regret >= 20)"], rows)
                + "\n\n*Caveat:* the reference is itself a Minimax search with the same evaluation, so it favours "
                  "Minimax-style choices over MCTS. Treat this as a measure of search quality, not of absolute strength.")
    names = [n for n, _ in agents]
    rep.chart("quality.png", lambda plt: _plot_bars(plt, names, [statistics.mean(regrets[n]) for n in names],
                                                    "Mean regret (lower = better)", "Decision quality", rot=30))


# ============================================================== strength
def exp_strength(rep, specs, games, openings, max_plies, seed):
    """Round-robin between every algorithm; reuses the rules of tournament.py."""
    import itertools
    import tournament as T

    names = {s: T.label(s) for s in specs}
    stat = {s: dict(w=0, d=0, l=0, t=0.0, m=0, walls=0, g=0) for s in specs}
    pair_rows, csv_rows = [], []
    t_start = time.time()
    n_sig = 0
    for a, b in itertools.combinations(specs, 2):
        wa = wb = dr = plies_total = 0
        for g in range(games):
            sd = seed * 1000 + g // 2
            start = T.opening(sd, openings)
            ap = g % 2                                   # colour swap inside each pair of games
            ag = {ap: T.make_agent(a, sd + 1), 1 - ap: T.make_agent(b, sd + 2)}
            winner, plies, times, moves, walls = T.play(ag, start, max_plies)
            plies_total += plies
            for spec, pl in ((a, ap), (b, 1 - ap)):
                st = stat[spec]
                st["t"] += times[pl]
                st["m"] += moves[pl]
                st["walls"] += walls[pl]
                st["g"] += 1
            if winner is None:
                dr += 1
                stat[a]["d"] += 1
                stat[b]["d"] += 1
            elif winner == ap:
                wa += 1
                stat[a]["w"] += 1
                stat[b]["l"] += 1
            else:
                wb += 1
                stat[b]["w"] += 1
                stat[a]["l"] += 1
        decisive = wa + wb
        p = binom_p(wa, decisive) if decisive else 1.0
        n_sig += p < 0.05
        pair_rows.append([names[a], names[b], f"{wa}-{dr}-{wb}", f"{100 * (wa + dr / 2) / games:.0f}%",
                          f"{p:.3f}", "yes" if p < 0.05 else "no", f"{plies_total / games:.0f}"])
        csv_rows.append([names[a], names[b], wa, dr, wb, p])
        print(f"   {names[a]:>20} vs {names[b]:<20} {wa}-{dr}-{wb}   [{time.time() - t_start:.0f}s]", flush=True)

    order = sorted(specs, key=lambda s: -(stat[s]["w"] + 0.5 * stat[s]["d"]))
    rows = []
    for s_ in order:
        st = stat[s_]
        score = st["w"] + 0.5 * st["d"]
        lo, hi = wilson(score, st["g"])
        rows.append([names[s_], st["g"], st["w"], st["d"], st["l"], f"{100 * score / st['g']:.0f}%",
                     f"{100 * lo:.0f}-{100 * hi:.0f}%", fmt(1000 * st["t"] / max(1, st["m"]), 1),
                     fmt(st["walls"] / st["g"], 1)])
    write_csv(os.path.join(rep.out, "strength_pairs.csv"),
              ["agent_a", "agent_b", "a_wins", "draws", "b_wins", "p_value"], csv_rows)
    write_csv(os.path.join(rep.out, "strength_standings.csv"),
              ["agent", "games", "wins", "draws", "losses", "ms_per_move", "walls_per_game"],
              [[names[s_], stat[s_]["g"], stat[s_]["w"], stat[s_]["d"], stat[s_]["l"],
                1000 * stat[s_]["t"] / max(1, stat[s_]["m"]), stat[s_]["walls"] / stat[s_]["g"]] for s_ in order])
    body = (f"Round-robin, {games} games per pair, colour-swapped from the same {openings}-ply random openings, "
            f"max {max_plies} plies (a game that is not finished is a draw). Strength is set by depth / iterations, "
            "so only the time-per-move column depends on your machine.\n\n**Standings**\n\n"
            + md_table(["Agent", "Games", "W", "D", "L", "Score %", "95% CI", "ms / move", "walls / game"], rows)
            + "\n\n**Head-to-head** (p = exact binomial test on decisive games, H0: equal strength)\n\n"
            + md_table(["Agent A", "Agent B", "A-D-B", "A score", "p", "Significant?", "Avg plies"], pair_rows))
    rep.section("8. Playing strength: round-robin tournament", body)
    rep.chart("strength.png", lambda plt: _plot_bars(
        plt, [names[s_] for s_ in order],
        [100 * (stat[s_]["w"] + 0.5 * stat[s_]["d"]) / stat[s_]["g"] for s_ in order],
        "Score % (win = 1, draw = 0.5)", "Round-robin results", rot=25))
    rep.note(f"{n_sig} of {len(pair_rows)} pairings are statistically significant at the 5% level; "
             "any pairing marked 'no' is a tie as far as these games can tell. Increase --games to separate close pairs.")


# ================================================================== stats
def exp_stats(rep):
    rows = [result_row(*r) for r in KNOWN_RESULTS]
    need = next(w for w in range(21, 41) if binom_p(w, 40) < 0.05)
    rep.section("7. Are your recorded tournament results significant?",
                "Exact two-sided binomial test against 'both players equally strong' (p) and Wilson 95% interval of the "
                "win rate. Elo difference is the rating gap implied by the win rate.\n\n"
                + md_table(["Result", "Wins/games", "Win rate", "95% CI", "p-value", "Elo diff"], rows)
                + "\n\nA result is significant at the 5% level only when p < 0.05 (the interval then excludes 50%). "
                  f"With 40 games, {need} wins ({100 * need / 40:.0f}%) are needed for p < 0.05, so 62% is within noise. "
                  "Use `python analysis.py --match WINS LOSSES` for new results.")


# ================================================================= plots
def _plot_lines(plt, xs, series, labels, xl, yl, title):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, ys in series.items():
        ax.plot(xs, ys, marker="o", label=labels[k])
    ax.set_xlabel(xl), ax.set_ylabel(yl), ax.set_title(title), ax.legend(), ax.grid(alpha=.3)
    return fig


def _plot_bars(plt, names, vals, yl, title, rot=0, log=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(range(len(names)), vals, color="#4c78a8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=rot, ha="right" if rot else "center", fontsize=8)
    ax.set_ylabel(yl), ax.set_title(title), ax.grid(axis="y", alpha=.3)
    if log:
        ax.set_yscale("log")
    fig.tight_layout()
    return fig


def _plot_xy(plt, xs, ys, xl, yl, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xl), ax.set_ylabel(yl), ax.set_title(title), ax.grid(alpha=.3)
    return fig


def _plot_depth(plt, depths, per):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))
    a.semilogy(depths, [statistics.mean(x[1] for x in per[d]) for d in depths], marker="o")
    a.set_xlabel("Search depth"), a.set_ylabel("Mean nodes (log)"), a.set_title("Nodes vs depth"), a.grid(alpha=.3)
    b.semilogy(depths, [statistics.mean(x[2] for x in per[d]) * 1000 for d in depths], marker="o", color="tab:red")
    b.set_xlabel("Search depth"), b.set_ylabel("Mean time (ms, log)"), b.set_title("Time vs depth"), b.grid(alpha=.3)
    fig.tight_layout()
    return fig


# ================================================================== report
class Report:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.parts = ["# Quoridor algorithm performance analysis\n",
                      "Generated by `analysis.py`. Timings depend on the machine and Python version; "
                      "node counts do not.\n"]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self.plt = plt
        except ImportError:
            self.plt = None

    def section(self, title, body):
        self.parts.append(f"\n## {title}\n\n{body}\n")
        print(f"[done] {title}")

    def note(self, text):
        self.parts.append(f"\n**Finding:** {text}\n")

    def chart(self, name, build):
        if self.plt is None:
            return
        fig = build(self.plt)
        fig.savefig(os.path.join(self.out, name), dpi=140)
        self.plt.close(fig)
        self.parts.append(f"\n![{name}]({name})\n")

    def save(self):
        path = os.path.join(self.out, "analysis_results.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.parts))
        print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all", help="comma list of: astar,branching,depth,ablation,mcts,quality,strength,stats")
    ap.add_argument("--quick", action="store_true", help="fewer positions and shallower searches")
    ap.add_argument("--positions", type=int, default=None, help="number of test positions")
    ap.add_argument("--max-depth", type=int, default=None, help="deepest Minimax depth for the depth study")
    ap.add_argument("--out", default="analysis_output")
    ap.add_argument("--agents", nargs="+", default=None,
                    help="strength experiment line-up, e.g. greedy minimax-basic:2 minimax:3 mcts:400")
    ap.add_argument("--games", type=int, default=None, help="games per pair in the strength experiment (even)")
    ap.add_argument("--openings", type=int, default=2, help="random opening plies for the tournament")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match", nargs=2, type=int, metavar=("WINS", "LOSSES"),
                    help="just print significance for one result and exit")
    a = ap.parse_args()

    if a.match:
        w, l = a.match
        print(md_table(["Result", "Wins/games", "Win rate", "95% CI", "p-value", "Elo diff"],
                       [result_row("your result", w, w + l)]))
        return

    wanted = set(a.exp.split(",")) if a.exp != "all" else {"astar", "branching", "depth", "ablation", "mcts", "quality", "strength", "stats"}
    npos = a.positions or (8 if a.quick else 24)
    dmax = a.max_depth or (3 if a.quick else 4)
    rep = Report(a.out)
    positions = sample_positions(npos) if wanted & {"branching", "depth", "ablation", "mcts", "quality"} else []
    if "astar" in wanted:
        exp_astar(rep, a.quick)
    if "branching" in wanted:
        exp_branching(rep, positions)
    if "depth" in wanted:
        exp_depth(rep, positions, list(range(1, dmax + 1)))
    if "ablation" in wanted:
        exp_ablation(rep, positions, 3)
    if "mcts" in wanted:
        exp_mcts(rep, positions, [100, 300, 1000] if a.quick else [100, 300, 1000, 2500])
    if "quality" in wanted:
        exp_quality(rep, positions, 3 if a.quick else 4,
                    [100, 300] if a.quick else [100, 300, 1000], [1, 2] if a.quick else [1, 2, 3])
    if "strength" in wanted:
        specs = a.agents or (["greedy", "minimax:2", "mcts:100"] if a.quick else
                             ["random", "greedy", "minimax-basic:2", "minimax:2", "minimax:3", "mcts:400"])
        exp_strength(rep, specs, a.games or (4 if a.quick else 8), a.openings, a.max_plies, a.seed)
    if "stats" in wanted:
        exp_stats(rep)
    rep.save()


if __name__ == "__main__":
    main()
