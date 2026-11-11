"""AI players for Quoridor.

MinimaxAgent
------------
Minimax with alpha-beta pruning and iterative deepening (time limited).
Two configurations share the same search code so they can be compared:

  improved=False  ("basic", the original agent)
      score = 10 * (opp_path - my_path) + 1.5 * (my_walls - opp_walls)
      walls: every wall crossing the opponent's A* path that lengthens it.

  improved=True   (default)
      * evaluation = path difference + tempo (side to move is half a step
        ahead) + wall difference + "flexibility" (number of optimal first
        steps; a pawn in a one-wide corridor is easy to block) + an exact
        race rule (a player with no walls left cannot be stopped, so the
        side that arrives first has won).
      * transposition table (kept between moves) with exact/lower/upper bound
        flags, and the stored best move is searched first.
      * walls are ranked by net gain (opponent lengthened minus self
        lengthened) and only the best `max_walls` are searched.
      * ties at the root are broken randomly, so games are not all identical.

MCTS lives in mcts.py.  GreedyAgent / RandomAgent are weak baselines.
"""
from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

from .game import State

WIN_SCORE = 10_000
RACE_SCORE = 2_000
W_PATH = 10.0
W_WALL = 1.5            # original weight (basic agent)
IMPROVED_WEIGHTS = {"wall": 2.5, "tempo": 4.0, "flex": 0.5, "race": 1.0}

# level -> (max search depth, time limit in seconds per move)
LEVELS = {1: (1, 0.5), 2: (2, 1.5), 3: (3, 3.0), 4: (4, 6.0)}

_EXACT, _LOWER, _UPPER = 0, 1, 2
_TT_LIMIT = 400_000


class SearchTimeout(Exception):
    pass


class Agent:
    name = "agent"

    def choose_move(self, state: State):
        raise NotImplementedError


class RandomAgent(Agent):
    name = "Random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_move(self, state: State):
        pawn = [("move", r, c) for r, c in state.pawn_moves()]
        if state.walls_left[state.turn] > 0 and self.rng.random() < 0.3:
            walls = state.legal_walls()
            if walls:
                return self.rng.choice(walls)
        return self.rng.choice(pawn)


class GreedyAgent(Agent):
    """Always takes the pawn move that gets it closest to the goal (A*)."""
    name = "Greedy A*"

    def choose_move(self, state: State):
        p = state.turn
        best, best_d = None, 10 ** 9
        for r, c in state.pawn_moves():
            mv = ("move", r, c)
            child = state.apply(mv, check=False)
            d = 0 if child.winner() == p else child.dist(p)
            if d < best_d:
                best, best_d = mv, d
        return best


# ---------------------------------------------------------------- wall logic
def candidate_walls(state: State, target: int):
    """Geometrically valid walls that cross an edge of `target`'s shortest path.

    A wall that does not cross the current shortest path cannot lengthen it,
    so this set contains every wall that can possibly hurt `target`.
    """
    path = state.shortest_path(target)
    seen, out = set(), []
    for a, b in zip(path, path[1:]):
        if a[0] != b[0]:                       # vertical step -> horizontal walls
            r, c = min(a[0], b[0]), a[1]
            options = ((r, c - 1, "H"), (r, c, "H"))
        else:                                  # horizontal step -> vertical walls
            r, c = a[0], min(a[1], b[1])
            options = ((r - 1, c, "V"), (r, c, "V"))
        for w in options:
            if w not in seen and state.wall_fits(*w):
                seen.add(w)
                out.append(w)
    return out


# ---------------------------------------------------------------- evaluation
def basic_evaluate(state: State, me: int) -> float:
    w = state.winner()
    if w is not None:
        return WIN_SCORE if w == me else -WIN_SCORE
    opp = 1 - me
    return (W_PATH * (state.dist(opp) - state.dist(me))
            + W_WALL * (state.walls_left[me] - state.walls_left[opp]))


def improved_evaluate(state: State, me: int, weights=IMPROVED_WEIGHTS) -> float:
    w = state.winner()
    if w is not None:
        return WIN_SCORE if w == me else -WIN_SCORE
    opp = 1 - me
    d_me, d_opp = state.dist(me), state.dist(opp)
    wl_me, wl_opp = state.walls_left[me], state.walls_left[opp]
    me_to_move = state.turn == me
    path_term = W_PATH * (d_opp - d_me)

    # Race rule: a player without walls cannot slow the other one down, so if
    # the other player is also ahead in the race, the game is decided.
    if weights["race"] and (wl_opp == 0 or wl_me == 0):
        me_first = d_me <= d_opp if me_to_move else d_me < d_opp
        if wl_opp == 0 and me_first:
            return RACE_SCORE + path_term
        if wl_me == 0 and not me_first:
            return -RACE_SCORE + path_term

    return (path_term
            + (weights["tempo"] if me_to_move else -weights["tempo"])
            + weights["wall"] * (wl_me - wl_opp)
            + weights["flex"] * (state.options(me) - state.options(opp)))


class MinimaxAgent(Agent):
    def __init__(self, max_depth: int = 3, time_limit: float = 3.0, useful_only: bool = True,
                 improved: bool = True, max_walls: int = 10, seed: Optional[int] = None,
                 weights: Optional[dict] = None, use_tt: bool = True):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.useful_only = useful_only
        self.improved = improved
        self.use_tt = improved and use_tt
        self.max_walls = max_walls
        self.weights = dict(IMPROVED_WEIGHTS, **(weights or {}))
        self.rng = random.Random(seed)
        self.name = f"Minimax(d={max_depth}{'' if improved else ',basic'})"
        self.last_stats = {}
        self._deadline = 0.0
        self._nodes = 0
        self._tt: dict = {}

    # ------------------------------------------------------------ evaluation
    def evaluate(self, state: State, me: int) -> float:
        if self.improved:
            return improved_evaluate(state, me, self.weights)
        return basic_evaluate(state, me)

    # ------------------------------------------------------- move generation
    def _children(self, state: State, me: int, order: bool, first=None) -> List[Tuple[tuple, State]]:
        p = state.turn
        kids = []
        for r, c in state.pawn_moves(p):
            mv = ("move", r, c)
            kids.append((mv, state.apply(mv, check=False)))
        if state.walls_left[p] > 0:
            opp = 1 - p
            base_opp, base_me = state.dist(opp), state.dist(p)
            walls = []
            for r, c, o in candidate_walls(state, opp):
                child = state.try_wall(r, c, o)
                if child is None:
                    continue
                gain = child.dist(opp) - base_opp
                if self.useful_only and gain <= 0:
                    continue
                net = gain - (child.dist(p) - base_me) if self.improved else 0
                walls.append((net, ("wall", r, c, o), child))
            if self.improved:                       # best walls first, cap the rest
                walls.sort(key=lambda x: -x[0])
                walls = walls[:self.max_walls]
            kids.extend((mv, ch) for _, mv, ch in walls)
        if order:
            kids.sort(key=lambda mc: self.evaluate(mc[1], me), reverse=(p == me))
        if first is not None:                        # transposition-table move first
            for i, (mv, _) in enumerate(kids):
                if mv == first:
                    kids.insert(0, kids.pop(i))
                    break
        return kids

    # ---------------------------------------------------------------- search
    def _minimax(self, state: State, depth: int, alpha: float, beta: float, me: int) -> float:
        self._nodes += 1
        if time.time() > self._deadline:
            raise SearchTimeout
        w = state.winner()
        if w is not None:                          # earlier win/loss is better/worse
            return (WIN_SCORE + depth) if w == me else -(WIN_SCORE + depth)
        if depth == 0:
            return self.evaluate(state, me)

        key, tt_move = None, None
        if self.use_tt:
            key = (state.pawns, state.walls_left, state.walls, state.turn, me)
            entry = self._tt.get(key)
            if entry is not None:
                e_depth, flag, val, tt_move = entry
                if e_depth >= depth:
                    if flag == _EXACT:
                        return val
                    if flag == _LOWER:
                        alpha = max(alpha, val)
                    else:
                        beta = min(beta, val)
                    if alpha >= beta:
                        return val

        a0, b0 = alpha, beta
        kids = self._children(state, me, order=depth > 1, first=tt_move)
        best_mv = None
        if state.turn == me:                       # MAX node
            best = float("-inf")
            for mv, child in kids:
                v = self._minimax(child, depth - 1, alpha, beta, me)
                if v > best:
                    best, best_mv = v, mv
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:                                      # MIN node
            best = float("inf")
            for mv, child in kids:
                v = self._minimax(child, depth - 1, alpha, beta, me)
                if v < best:
                    best, best_mv = v, mv
                beta = min(beta, best)
                if alpha >= beta:
                    break

        if key is not None:
            if len(self._tt) > _TT_LIMIT:
                self._tt.clear()
            flag = _UPPER if best <= a0 else _LOWER if best >= b0 else _EXACT
            self._tt[key] = (depth, flag, best, best_mv)
        return best

    def choose_move(self, state: State):
        t0 = time.time()
        self._deadline = t0 + self.time_limit
        self._nodes = 0
        me = state.turn
        kids = self._children(state, me, order=True)
        best_move, best_val, reached = kids[0][0], None, 0

        for depth in range(1, self.max_depth + 1):
            scored = []
            top = float("-inf")
            try:
                for mv, child in kids:
                    # improved: search just below the best value so equal moves are exact
                    alpha = top - 1e-6 if self.improved else top
                    val = self._minimax(child, depth - 1, alpha, float("inf"), me)
                    scored.append((val, mv, child))
                    top = max(top, val)
            except SearchTimeout:
                break
            scored.sort(key=lambda x: -x[0])       # stable: best first for next depth
            kids = [(mv, ch) for _, mv, ch in scored]
            best_val, reached = scored[0][0], depth
            if self.improved:
                ties = [mv for v, mv, _ in scored if v == best_val]
                best_move = self.rng.choice(ties)
            else:
                best_move = scored[0][1]
            if abs(best_val) >= WIN_SCORE:         # forced win/loss found
                break

        self.last_stats = {"depth": reached, "nodes": self._nodes,
                           "time": time.time() - t0, "score": best_val}
        return best_move
