"""Path-finding helpers for Quoridor.

A* is used for two things:
  1. Movement planning  - the shortest route from a pawn to its goal row.
  2. Rule checking      - a wall is illegal if it leaves a player with no route.
  3. Heuristic          - the evaluation function of the Minimax agent is built
                          on the A* shortest-path lengths of both pawns.

The goal is a whole row, not a single cell, so the heuristic is simply the
number of rows still to cross.  It never over-estimates (every step changes the
row by at most one), hence A* returns an optimal path.

Pawns are ignored while searching (the standard definition of "distance" in
Quoridor AI): jumping over the opponent is a tactical detail that would make
the distance flicker from move to move.
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Optional, Tuple

Cell = Tuple[int, int]


def astar(state, start: Cell, goal_row: int) -> Optional[Tuple[Cell, ...]]:
    """Return the shortest path (tuple of cells, start included) or None.

    `state` only needs a `neighbors(cell)` method that yields reachable cells
    (i.e. it already takes walls into account).
    """
    open_heap = [(abs(start[0] - goal_row), 0, start)]
    best_g = {start: 0}
    parent = {}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur[0] == goal_row:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return tuple(path)
        closed.add(cur)
        for nb in state.neighbors(cur):
            ng = g + 1
            if ng < best_g.get(nb, 1 << 30):
                best_g[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_heap, (ng + abs(nb[0] - goal_row), ng, nb))
    return None


def bfs_distance(state, start: Cell, goal_row: int) -> Optional[int]:
    """Plain breadth-first search; used by the tests to cross-check A*."""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cur, d = queue.popleft()
        if cur[0] == goal_row:
            return d
        for nb in state.neighbors(cur):
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, d + 1))
    return None