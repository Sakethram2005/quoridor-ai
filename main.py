"""Entry point: python main.py [--mode MODE] [--level 1-4]

Modes:  human-ai (default) | demo (Minimax vs Greedy) | ai-ai | human-human
"""
import argparse

MODE_INDEX = {"human-ai": 0, "demo": 1, "ai-ai": 2, "human-human": 3}


def main():
    ap = argparse.ArgumentParser(description="Quoridor with an A* + Minimax AI")
    ap.add_argument("--mode", choices=MODE_INDEX, default="human-ai")
    ap.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=2,
                    help="AI strength: search depth 1-4 (default 2)")
    args = ap.parse_args()
    from quoridor.gui import run          # imported late so tests don't need pygame
    run(MODE_INDEX[args.mode], args.level)


if __name__ == "__main__":
    main()