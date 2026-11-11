"""Monte Carlo Tree Search (UCT) agent for Quoridor.

Pure random playouts are hopeless in Quoridor (a random pawn almost never
reaches the goal), so this agent uses informed playouts:

  * Selection       UCT  = mean reward + c * sqrt(ln(N_parent) / N_child)
  * Expansion       progressive widening: a node may only have about
                    1.5*sqrt(visits) children, and its untried moves are
                    ordered best-first (pawn moves that shorten the path, then
                    walls on the opponent's A* path, then the other pawn moves).
                    Walls are checked lazily: illegal ones, or ones that do not
                    lengthen the opponent's path, are discarded on expansion.
  * Simulation      each player walks its A* shortest path; with probability
                    `wall_prob` it instead places a random wall on the
                    opponent's path.  After `rollout_depth` plies the playout
                    is cut off and scored with a logistic function of the
                    path-length difference (so a rollout never returns noise).
  * Backpropagation reward in [0, 1] for the player who made the move.

The final move is the most-visited child of the root.
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional

from .agent import Agent, candidate_walls
from .game import GOAL_ROW, State


class _Node:
    __slots__ = ("state", "parent", "move", "mover", "children", "untried", "visits", "value")

    def __init__(self, state: State, parent=None, move=None, mover=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.mover = mover            # player who made `move` (None at the root)
        self.children = []
        self.untried = None           # built lazily, best moves first
        self.visits = 0
        self.value = 0.0              # total reward for `mover`


class MCTSAgent(Agent):
    def __init__(self, iterations: int = 400, time_limit: Optional[float] = None,
                 c: float = 0.8, rollout_depth: int = 30, wall_prob: float = 0.25,
                 temperature: float = 2.5, seed: Optional[int] = None):
        self.iterations = iterations
        self.time_limit = time_limit
        self.c = c
        self.rollout_depth = rollout_depth
        self.wall_prob = wall_prob
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.name = f"MCTS(n={iterations})"
        self.last_stats = {}

    # ------------------------------------------------------------- expansion
    def _ordered_moves(self, state: State):
        p, opp = state.turn, 1 - state.turn
        field = state.dist_field(p)
        pawn = sorted(state.pawn_moves(p), key=lambda cell: field.get(cell, 99))
        best_d = field.get(pawn[0], 99)
        good = [("move", r, c) for r, c in pawn if field.get((r, c), 99) <= best_d]
        rest = [("move", r, c) for r, c in pawn if field.get((r, c), 99) > best_d]
        walls = []
        if state.walls_left[p] > 0:
            walls = [("wall", r, c, o) for r, c, o in candidate_walls(state, opp)]
        return good + walls + rest

    def _expand(self, node: _Node):
        """Pop untried moves until one is playable; return the new child or None."""
        s = node.state
        p, opp = s.turn, 1 - s.turn
        while node.untried:
            mv = node.untried.pop(0)
            if mv[0] == "move":
                child = s.apply(mv, check=False)
            else:
                child = s.try_wall(mv[1], mv[2], mv[3])
                if child is None or child.dist(opp) <= s.dist(opp):
                    continue                      # illegal or pointless wall
            node.children.append(_Node(child, node, mv, p))
            return node.children[-1]
        return None

    # ------------------------------------------------------------ simulation
    def _rollout(self, state: State) -> float:
        """Return the (estimated) probability that player 0 wins."""
        s, rng = state, self.rng
        for _ in range(self.rollout_depth):
            w = s.winner()
            if w is not None:
                return 1.0 if w == 0 else 0.0
            p = s.turn
            nxt = None
            if s.walls_left[p] > 0 and rng.random() < self.wall_prob:
                cands = candidate_walls(s, 1 - p)
                if cands:
                    r, c, o = rng.choice(cands)
                    child = s.try_wall(r, c, o)
                    if child is not None and child.dist(1 - p) > s.dist(1 - p):
                        nxt = child
            if nxt is None:
                moves = s.pawn_moves(p)
                step = s.shortest_path(p)[1]
                if step not in moves:               # blocked by the other pawn: use jumps
                    field = s.dist_field(p)
                    step = min(moves, key=lambda cell: field.get(cell, 99))
                nxt = s.apply(("move", step[0], step[1]), check=False)
            s = nxt
        w = s.winner()
        if w is not None:
            return 1.0 if w == 0 else 0.0
        adv = (s.dist(1) - s.dist(0)) + (0.5 if s.turn == 0 else -0.5)
        return 1.0 / (1.0 + math.exp(-adv / self.temperature))

    # ---------------------------------------------------------------- search
    def _select(self, root: _Node):
        node = root
        while True:
            if node.state.winner() is not None:
                return node
            if node.untried is None:
                node.untried = self._ordered_moves(node.state)
            allowed = 2 + int(1.5 * math.sqrt(node.visits))
            if node.untried and len(node.children) < allowed:
                child = self._expand(node)
                if child is not None:
                    return child
            if not node.children:
                return node
            log_n = math.log(node.visits)
            node = max(node.children,
                       key=lambda ch: ch.value / ch.visits + self.c * math.sqrt(log_n / ch.visits))

    def choose_move(self, state: State):
        t0 = time.time()
        p = state.turn
        for r, c in state.pawn_moves(p):               # take an immediate win
            if r == GOAL_ROW[p]:
                self.last_stats = {"iterations": 0, "nodes": 1, "depth": 1,
                                   "time": 0.0, "score": 1.0}
                return ("move", r, c)

        root = _Node(state)
        root.untried = self._ordered_moves(state)
        deadline = t0 + self.time_limit if self.time_limit else None
        done = 0
        while done < self.iterations:
            if deadline is not None and done >= 8 and time.time() > deadline:
                break
            leaf = self._select(root)
            p0 = self._rollout(leaf.state)
            node = leaf
            while node is not None:
                node.visits += 1
                if node.mover is not None:
                    node.value += p0 if node.mover == 0 else 1.0 - p0
                node = node.parent
            done += 1

        best = max(root.children, key=lambda ch: (ch.visits, ch.value))
        depth, n, nodes = 0, best, 0
        while n.children:                              # depth of the principal variation
            n = max(n.children, key=lambda ch: ch.visits)
            depth += 1
        stack = [root]
        while stack:
            cur = stack.pop()
            nodes += 1
            stack.extend(cur.children)
        self.last_stats = {"iterations": done, "nodes": nodes, "depth": depth + 1,
                           "time": time.time() - t0, "score": best.value / best.visits}
        return best.move
