"""Quoridor rules engine (board, pawn moves, jumps, wall legality).

Coordinates
-----------
Cells are (row, col), row 0 = top, 9x9 board.
  * Player 0 starts at the bottom (8, 4) and must reach row 0.
  * Player 1 starts at the top    (0, 4) and must reach row 8.

Walls are (r, c, orientation) with r, c in 0..7 (they live on the 8x8 grid of
"corners" between cells).
  * "H" wall at (r, c): lies between row r and row r+1, spanning columns c, c+1.
  * "V" wall at (r, c): lies between col c and col c+1, spanning rows r, r+1.

Moves are tuples:
  ("move", r, c)          - move the pawn to cell (r, c)
  ("wall", r, c, "H"/"V") - place a wall

`State` is treated as immutable: `apply()` returns a new State.  This keeps the
Minimax code simple and safe (no undo bookkeeping).
"""
from __future__ import annotations

from typing import FrozenSet, List, Optional, Tuple

from . import pathfinding

N = 9
WALLS_PER_PLAYER = 10
GOAL_ROW = (0, N - 1)
START = ((N - 1, N // 2), (0, N // 2))
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

Cell = Tuple[int, int]
Move = tuple

# Shortest paths depend only on (walls, pawn position, player) -> memoise them.
_PATH_CACHE: dict = {}
_PATH_CACHE_LIMIT = 300_000


def wall_edges(r: int, c: int, o: str):
    """The two cell-to-cell edges a wall blocks."""
    if o == "H":
        return (((r, c), (r + 1, c)), ((r, c + 1), (r + 1, c + 1)))
    return (((r, c), (r, c + 1)), ((r + 1, c), (r + 1, c + 1)))


def _blocked_from_walls(walls) -> FrozenSet:
    out = set()
    for r, c, o in walls:
        for a, b in wall_edges(r, c, o):
            out.add((a, b))
            out.add((b, a))
    return frozenset(out)


class State:
    __slots__ = ("pawns", "walls_left", "walls", "blocked", "turn")

    def __init__(self, pawns=START, walls_left=(WALLS_PER_PLAYER, WALLS_PER_PLAYER),
                 walls=frozenset(), turn: int = 0, blocked=None):
        self.pawns = tuple(pawns)
        self.walls_left = tuple(walls_left)
        self.walls = frozenset(walls)
        self.turn = turn
        self.blocked = blocked if blocked is not None else _blocked_from_walls(self.walls)

    # ------------------------------------------------------------------ basics
    def winner(self) -> Optional[int]:
        for p in (0, 1):
            if self.pawns[p][0] == GOAL_ROW[p]:
                return p
        return None

    def _can_step(self, a: Cell, b: Cell) -> bool:
        return 0 <= b[0] < N and 0 <= b[1] < N and (a, b) not in self.blocked

    def neighbors(self, cell: Cell):
        """Adjacent cells reachable without crossing a wall (pawns ignored)."""
        r, c = cell
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if self._can_step(cell, nb):
                yield nb

    # ----------------------------------------------------------- shortest path
    def shortest_path(self, player: int):
        key = (self.walls, self.pawns[player], player)
        try:
            return _PATH_CACHE[key]
        except KeyError:
            pass
        if len(_PATH_CACHE) > _PATH_CACHE_LIMIT:
            _PATH_CACHE.clear()
        path = pathfinding.astar(self, self.pawns[player], GOAL_ROW[player])
        _PATH_CACHE[key] = path
        return path

    def dist(self, player: int) -> Optional[int]:
        path = self.shortest_path(player)
        return None if path is None else len(path) - 1

    # -------------------------------------------------------------- pawn moves
    def pawn_moves(self, player: Optional[int] = None) -> List[Cell]:
        """Legal destination cells, including straight and diagonal jumps."""
        if player is None:
            player = self.turn
        me, opp = self.pawns[player], self.pawns[1 - player]
        moves: List[Cell] = []
        for nb in self.neighbors(me):
            if nb != opp:
                moves.append(nb)
                continue
            dr, dc = nb[0] - me[0], nb[1] - me[1]
            far = (opp[0] + dr, opp[1] + dc)
            if self._can_step(opp, far):                 # straight jump
                moves.append(far)
            else:                                        # blocked -> diagonal jumps
                for pdr, pdc in ((dc, dr), (-dc, -dr)):
                    side = (opp[0] + pdr, opp[1] + pdc)
                    if self._can_step(opp, side):
                        moves.append(side)
        return moves

    # ------------------------------------------------------------------- walls
    def wall_fits(self, r: int, c: int, o: str) -> bool:
        """Geometry only: in bounds, no overlap, no crossing."""
        if not (0 <= r < N - 1 and 0 <= c < N - 1) or o not in ("H", "V"):
            return False
        w = self.walls
        if (r, c, o) in w or (r, c, "V" if o == "H" else "H") in w:
            return False
        if o == "H":
            return (r, c - 1, "H") not in w and (r, c + 1, "H") not in w
        return (r - 1, c, "V") not in w and (r + 1, c, "V") not in w

    def _place_wall(self, r: int, c: int, o: str, player: int) -> "State":
        new_blocked = set(self.blocked)
        for a, b in wall_edges(r, c, o):
            new_blocked.add((a, b))
            new_blocked.add((b, a))
        wl = list(self.walls_left)
        wl[player] -= 1
        return State(self.pawns, wl, self.walls | {(r, c, o)}, 1 - player,
                     frozenset(new_blocked))

    def try_wall(self, r: int, c: int, o: str, player: Optional[int] = None) -> Optional["State"]:
        """Place a wall if fully legal (incl. 'both players keep a path')."""
        if player is None:
            player = self.turn
        if self.walls_left[player] <= 0 or not self.wall_fits(r, c, o):
            return None
        new = self._place_wall(r, c, o, player)
        if new.shortest_path(0) is None or new.shortest_path(1) is None:
            return None
        return new

    # ------------------------------------------------------------ move helpers
    def is_legal(self, move: Move) -> bool:
        if move[0] == "move":
            return (move[1], move[2]) in self.pawn_moves(self.turn)
        if move[0] == "wall":
            return self.try_wall(move[1], move[2], move[3]) is not None
        return False

    def apply(self, move: Move, check: bool = True) -> "State":
        p = self.turn
        if move[0] == "move":
            cell = (move[1], move[2])
            if check and cell not in self.pawn_moves(p):
                raise ValueError(f"illegal pawn move {move}")
            pawns = list(self.pawns)
            pawns[p] = cell
            return State(pawns, self.walls_left, self.walls, 1 - p, self.blocked)
        if move[0] == "wall":
            _, r, c, o = move
            new = self.try_wall(r, c, o) if check else self._place_wall(r, c, o, p)
            if new is None:
                raise ValueError(f"illegal wall {move}")
            return new
        raise ValueError(f"unknown move {move}")

    def legal_walls(self) -> List[Move]:
        out = []
        if self.walls_left[self.turn] <= 0:
            return out
        for r in range(N - 1):
            for c in range(N - 1):
                for o in ("H", "V"):
                    if self.try_wall(r, c, o) is not None:
                        out.append(("wall", r, c, o))
        return out

    def legal_moves(self) -> List[Move]:
        return [("move", r, c) for r, c in self.pawn_moves()] + self.legal_walls()

    # ------------------------------------------------------------------- debug
    def __str__(self) -> str:
        """Small ASCII rendering (handy when debugging in a terminal)."""
        rows = []
        for r in range(N):
            line = ""
            for c in range(N):
                ch = "A" if self.pawns[0] == (r, c) else "B" if self.pawns[1] == (r, c) else "."
                line += ch
                if c < N - 1:
                    line += "|" if ((r, c), (r, c + 1)) in self.blocked else " "
            rows.append(line)
            if r < N - 1:
                sep = ""
                for c in range(N):
                    sep += "-" if ((r, c), (r + 1, c)) in self.blocked else " "
                    if c < N - 1:
                        sep += " "
                rows.append(sep)
        return "\n".join(rows)


def cell_name(r: int, c: int) -> str:
    """Chess-like name: column a-i, row 1-9 counted from the bottom."""
    return f"{'abcdefghi'[c]}{N - r}"


def describe_move(move: Move) -> str:
    if move[0] == "move":
        return f"moves to {cell_name(move[1], move[2])}"
    _, r, c, o = move
    return f"places {o}-wall at {cell_name(r, c)}"