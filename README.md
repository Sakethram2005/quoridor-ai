# Quoridor AI - A* path-finding + Minimax (alpha-beta) with a pygame demo

An agent that plays Quoridor against a human or another AI. Reach the opposite
row first; on each turn either move your pawn or place one of your 10 walls.

## Project layout

```
quoridor_project/
├── main.py                 # entry point for the pygame demo
├── tournament.py           # headless AI-vs-AI evaluation (for the report)
├── requirements.txt        # pygame
├── quoridor/
│   ├── __init__.py
│   ├── game.py             # rules: board, pawn moves + jumps, wall legality
│   ├── pathfinding.py      # A* (and BFS used only to verify A* in tests)
│   ├── agent.py            # MinimaxAgent (+ Greedy and Random baselines)
│   └── gui.py              # pygame interface
└── tests/
    └── test_quoridor.py    # unit tests (rules, A*, agent)
```

## Run

```bash
pip install -r requirements.txt
python main.py                      # Human (blue, bottom) vs Minimax AI (red, top)
python main.py --mode demo          # Minimax vs Greedy - best for the live demo
python main.py --mode ai-ai --level 3
python main.py --level 4            # strongest / slowest AI
```

Controls: click a green dot to move your pawn; hover near a groove between
cells to preview a wall (green = legal, red = illegal) and click to place it.
`N` new game · `M` change mode · `L` change AI level · `P` show A* paths · `Esc` quit.
Turning on the A* paths (`P`) is a good way to show the examiners what the
algorithm "sees".

Tests and evaluation (no pygame needed):

```bash
python -m unittest discover -v
python tournament.py --games 10 --depth 3 --opponent greedy
python tournament.py --games 10 --depth 3 --opponent minimax --opp-depth 1
```

## How it works

**State / rules (`game.py`).** Immutable `State` (pawns, walls left, walls,
blocked edges, side to move). Pawn moves include straight jumps and diagonal
jumps when a wall or the board edge is behind the opponent. A wall is legal
only if it is in bounds, does not overlap or cross another wall, the player has
walls left, and **both players still have a path to their goal**.

**A* (`pathfinding.py`).** Goal = any cell of the target row, heuristic = rows
remaining (admissible and consistent), so paths are optimal. It is used for
(1) the shortest-path length of each pawn, (2) the "wall must not seal a
player in" rule, (3) the Minimax evaluation, (4) the Greedy baseline and the
path overlay in the GUI. Results are memoised on (walls, pawn cell, player).

**Minimax (`agent.py`).**
* Minimax with alpha-beta pruning and iterative deepening under a per-move
  time limit (levels 1-4 = depth 1-4).
* Evaluation from the searching player's view:
  `10 * (opp_distance - my_distance) + 1.5 * (my_walls - opp_walls)`;
  terminal win/loss = ±10000 (+ remaining depth so faster wins are preferred).
* Branching control: only *useful* walls are generated, i.e. walls crossing the
  opponent's current A* path that really lengthen it. Pawn moves are always
  generated. Children are ordered by static evaluation so alpha-beta prunes
  more.

**Possible extensions** (good for the report's future-work section): a
transposition table, wall candidates around the AI's own path to defend,
a smarter evaluation (e.g. mobility / pawn-adjacent walls), or comparing with
MCTS.

## Suggested report experiments

| Experiment | Command |
|---|---|
| Minimax vs Random / Greedy | `python tournament.py --opponent random` |
| Effect of depth | `python tournament.py --depth 3 --opponent minimax --opp-depth 1` |
| Effect of wall pruning | set `useful_only=False` in `MinimaxAgent` and compare time/nodes |