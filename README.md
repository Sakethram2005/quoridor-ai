# Quoridor AI - A* path-finding + Minimax (alpha-beta) + MCTS, with a pygame demo

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
│   ├── agent.py            # MinimaxAgent (basic + improved), Greedy and Random baselines
│   ├── mcts.py             # MCTSAgent (UCT with heuristic rollouts)
│   ├── players.py          # player choices (Human / Minimax L1-4 / MCTS L1-4 / Greedy / Random)
│   └── gui.py              # pygame interface with per-side dropdowns
└── tests/
    ├── test_quoridor.py    # unit tests (rules, A*, agents)
    └── test_players.py     # unit tests for the player options
```

## Run

```bash
pip install -r requirements.txt
python main.py                      # then choose Blue and Red in the dropdowns
python main.py --blue minimax4 --red minimax2   # start with Level 4 vs Level 2
python main.py --mode demo          # Minimax vs Greedy - best for the live demo
python main.py --mode minimax-mcts --level 3    # watch Minimax vs MCTS
```

Each side (Blue = bottom, Red = top) has its own dropdown in the side panel:
**Human, Minimax L1-L4 (depth 1-4), MCTS L1-L4 (150 / 400 / 1000 / 2500
iterations), Greedy A\*, Random**. Any combination works, e.g. Minimax L4 vs
Minimax L2. Changing a dropdown restarts the game. Command-line keys for
`--blue/--red`: `human`, `minimax1`-`minimax4`, `mcts1`-`mcts4`, `greedy`, `random`.

Controls: click a green dot to move your pawn; hover near a groove between
cells to preview a wall (green = legal, red = illegal) and click to place it.
`N` new game · `S` swap sides · `P` show A* paths · `Esc` quit.
Turning on the A* paths (`P`) is a good way to show the examiners what the
algorithm "sees".

Tests and evaluation (no pygame needed):

```bash
python -m unittest discover -v
python tournament.py                                   # default round-robin
python tournament.py --agents greedy minimax-basic:2 minimax:2 mcts:600 --games 10
```

## How it works

**State / rules (`game.py`).** Immutable `State` (pawns, walls left, walls,
blocked edges, side to move). Pawn moves include straight jumps and diagonal
jumps when a wall or the board edge is behind the opponent. A wall is legal
only if it is in bounds, does not overlap or cross another wall, the player has
walls left, and **both players still have a path to their goal**.

**A\* (`pathfinding.py`).** Goal = any cell of the target row, heuristic = rows
remaining (admissible and consistent), so paths are optimal. It gives (1) the
shortest-path length of each pawn, (2) the "wall must not seal a player in"
rule, (3) the evaluation, (4) the Greedy baseline, the MCTS playouts and the GUI
path overlay. Results are memoised on (walls, pawn cell, player), and stepping
along a path re-uses the rest of that path. A reverse BFS (`distance_field`)
gives the distance of every cell at once; it is used for the flexibility feature
and for cheap move ordering.

**Minimax (`agent.py`)** - alpha-beta + iterative deepening under a time limit
(levels 1-4 = depth 1-4). It exists in two configurations on the same search
code, so the improvements can be measured:

| | basic (`improved=False`) | improved (default) |
|---|---|---|
| Evaluation | `10*(opp_dist - my_dist) + 1.5*(my_walls - opp_walls)` | path difference + tempo (side to move = half a step) + wall difference + **flexibility** (number of optimal first steps: a pawn in a corridor is easy to block) + **race rule** (a player with no walls left cannot be stopped, so whoever arrives first has won) |
| Walls searched | every wall on the opponent's path that lengthens it | same, ranked by net gain (opponent lengthened minus self lengthened), best 10 kept |
| Move ordering | static evaluation | transposition-table move first, then static evaluation |
| Transposition table | no | yes (exact / lower / upper bounds, kept between moves) |
| Root ties | first best move | random among equal moves (games are not all identical) |

**MCTS (`mcts.py`)** - UCT with informed playouts, because random playouts never
finish a Quoridor game. Selection: UCT. Expansion: progressive widening
(~1.5*sqrt(visits) children) over best-first ordered moves; walls are checked
lazily and discarded if illegal or if they do not lengthen the opponent's path.
Simulation: each side follows its A* path, with a 25 % chance of placing a wall
on the opponent's path; after 30 plies the playout is cut off and scored with a
logistic function of the path-length difference. The most-visited root child is
played.

## Tournament

`tournament.py` plays a round-robin. Agent specs: `random`, `greedy`,
`minimax-basic:<depth>`, `minimax:<depth>`, `mcts:<iterations>`. Games come in
colour-swapped pairs that start from the same random opening (`--openings`
random pawn plies), because the player who moves first wins about 80 % of mirror
matches. Strength is controlled by depth / iterations rather than wall-clock
time, so results are reproducible; time per move is reported separately.
Results are written to `tournament_results.md`.

Possible extensions: wall candidates that also stop the opponent from blocking
you, a learned evaluation, opening book, parallel MCTS (root parallelisation).