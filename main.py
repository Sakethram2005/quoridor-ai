"""Entry point: python main.py [--mode MODE] [--level 1-4] [--blue KEY] [--red KEY]

Normally just run `python main.py` and pick who plays Blue and Red from the
dropdowns in the window (Human, Minimax L1-L4, MCTS L1-L4, Greedy, Random).

--mode / --level only choose the starting selection:
    human-ai (default) | demo (Minimax vs Greedy) | ai-ai | human-human
    | human-mcts | minimax-mcts
--blue / --red set a side directly, e.g.  --blue minimax4 --red minimax2
"""
import argparse

from quoridor.players import MODE_NAMES, PLAYER_KEYS


def main():
    ap = argparse.ArgumentParser(description="Quoridor with A* + Minimax / MCTS AIs")
    ap.add_argument("--mode", choices=MODE_NAMES, default="human-ai")
    ap.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=2,
                    help="AI strength 1-4 used by --mode (default 2)")
    ap.add_argument("--blue", choices=PLAYER_KEYS, help="player for Blue (bottom)")
    ap.add_argument("--red", choices=PLAYER_KEYS, help="player for Red (top)")
    args = ap.parse_args()
    from quoridor.gui import run          # imported late so tests don't need pygame
    run(args.mode, args.level, args.blue, args.red)


if __name__ == "__main__":
    main()