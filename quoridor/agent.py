"""AI players for Quoridor.

MinimaxAgent
------------
* Minimax with alpha-beta pruning and iterative deepening (time limited).
* Evaluation (from the point of view of the searching player `me`):

      score = W_PATH * (opp_shortest_path - my_shortest_path)
            + W_WALL * (my_walls_left - opp_walls_left)

  where the shortest paths are computed with A* (see pathfinding.py).
  A won position scores +WIN_SCORE (plus the remaining depth, so faster wins
  are preferred); a lost position scores the negative.
* Branching-factor control: a Quoridor position has up to ~130 legal moves, so
  only *useful* walls are generated: walls that block an edge of the opponent's
  current A* path AND really lengthen it.  Pawn moves are always generated.
* Move ordering (static evaluation of children) makes alpha-beta cut much more.

GreedyAgent / RandomAgent are weak baselines used for the demo and tournament.
"""
from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

from .game import State

WIN_SCORE = 10_000
W_PATH = 10.0
W_WALL = 1.5

# level -> (max search depth, time limit in seconds per move)
LEVELS = {1: (1, 0.5), 2: (2, 1.5), 3: (3, 3.0), 4: (4, 6.0)}


class SearchTimeout(Exception):
    pass


class Agent:
    name = "agent"

    def choose_move(self, state: State):
        raise NotImplementedError


class RandomAgent(Agent):
    name = "Random"

    def choose_move(self, state: State):
        pawn = [("move", r, c) for r, c in state.pawn_moves()]
        if state.walls_left[state.turn] > 0 and random.random() < 0.3:
            walls = state.legal_walls()
            if walls:
                return random.choice(walls)
        return random.choice(pawn)


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


class MinimaxAgent(Agent):
    def __init__(self, max_depth: int = 3, time_limit: float = 3.0, useful_only: bool = True):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.useful_only = useful_only
        self.name = f"Minimax(d={max_depth})"
        self.last_stats = {}
        self._deadline = 0.0
        self._nodes = 0

    # ------------------------------------------------------------ evaluation
    @staticmethod
    def evaluate(state: State, me: int) -> float:
        w = state.winner()
        if w is not None:
            return WIN_SCORE if w == me else -WIN_SCORE
        opp = 1 - me
        return (W_PATH * (state.dist(opp) - state.dist(me))
                + W_WALL * (state.walls_left[me] - state.walls_left[opp]))

    # ------------------------------------------------------- move generation
    @staticmethod
    def _candidate_walls(state: State, target: int):
        """Walls that cross an edge of `target`'s shortest path."""
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

    def _children(self, state: State, me: int, order: bool) -> List[Tuple[tuple, State]]:
        p = state.turn
        kids = []
        for r, c in state.pawn_moves(p):
            mv = ("move", r, c)
            kids.append((mv, state.apply(mv, check=False)))
        if state.walls_left[p] > 0:
            opp = 1 - p
            base = state.dist(opp)
            for r, c, o in self._candidate_walls(state, opp):
                child = state.try_wall(r, c, o)
                if child is None:
                    continue
                if self.useful_only and child.dist(opp) <= base:
                    continue
                kids.append((("wall", r, c, o), child))
        if order:
            kids.sort(key=lambda mc: self.evaluate(mc[1], me), reverse=(p == me))
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

        kids = self._children(state, me, order=depth > 1)
        if state.turn == me:                       # MAX node
            best = float("-inf")
            for _, child in kids:
                best = max(best, self._minimax(child, depth - 1, alpha, beta, me))
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        best = float("inf")                        # MIN node
        for _, child in kids:
            best = min(best, self._minimax(child, depth - 1, alpha, beta, me))
            beta = min(beta, best)
            if alpha >= beta:
                break
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
            alpha = float("-inf")
            try:
                for mv, child in kids:
                    val = self._minimax(child, depth - 1, alpha, float("inf"), me)
                    scored.append((val, mv, child))
                    alpha = max(alpha, val)
            except SearchTimeout:
                break
            scored.sort(key=lambda x: -x[0])       # stable: best first for next depth
            kids = [(mv, ch) for _, mv, ch in scored]
            best_val, best_move, reached = scored[0][0], scored[0][1], depth
            if abs(best_val) >= WIN_SCORE:         # forced win/loss found
                break

        self.last_stats = {"depth": reached, "nodes": self._nodes,
                           "time": time.time() - t0, "score": best_val}
        return best_move