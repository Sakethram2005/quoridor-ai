"""Player choices shared by the GUI and the command line (no pygame needed).

Each side (Blue / Red) can be set independently to a human or to any AI at
any strength, e.g. Minimax level 4 against Minimax level 2.
"""
from __future__ import annotations

from .agent import LEVELS, GreedyAgent, MinimaxAgent, RandomAgent
from .mcts import MCTSAgent

MCTS_ITERS = {1: 150, 2: 400, 3: 1000, 4: 2500}      # AI level -> MCTS iterations per move

# (key, label shown in the dropdown) - order = order in the dropdown
PLAYER_OPTIONS = (
    [("human", "Human")]
    + [(f"minimax{lv}", f"Minimax L{lv} (depth {LEVELS[lv][0]})") for lv in sorted(LEVELS)]
    + [(f"mcts{lv}", f"MCTS L{lv} ({MCTS_ITERS[lv]} iterations)") for lv in sorted(LEVELS)]
    + [("greedy", "Greedy A*"), ("random", "Random")]
)
PLAYER_KEYS = [k for k, _ in PLAYER_OPTIONS]
PLAYER_LABELS = dict(PLAYER_OPTIONS)

MODE_NAMES = ("human-ai", "demo", "ai-ai", "human-human", "human-mcts", "minimax-mcts")


def make_agent(key: str):
    """Fresh agent for a dropdown key; None means a human player."""
    if key == "human":
        return None
    if key == "greedy":
        return GreedyAgent()
    if key == "random":
        return RandomAgent()
    kind, level = key[:-1], int(key[-1])
    depth, time_limit = LEVELS[level]
    if kind == "minimax":
        return MinimaxAgent(depth, time_limit)
    if kind == "mcts":
        return MCTSAgent(MCTS_ITERS[level], time_limit=time_limit)
    raise ValueError(f"unknown player option {key!r}")


def preset_players(mode: str = "human-ai", level: int = 2):
    """(blue, red) keys for the old --mode names."""
    mm, mc = f"minimax{level}", f"mcts{level}"
    return {
        "human-ai": ("human", mm),
        "demo": (mm, "greedy"),
        "ai-ai": (mm, mm),
        "human-human": ("human", "human"),
        "human-mcts": ("human", mc),
        "minimax-mcts": (mm, mc),
    }[mode]