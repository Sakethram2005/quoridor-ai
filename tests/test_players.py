import unittest

from quoridor.agent import GreedyAgent, MinimaxAgent, RandomAgent
from quoridor.mcts import MCTSAgent
from quoridor.players import MODE_NAMES, PLAYER_KEYS, PLAYER_LABELS, make_agent, preset_players


class PlayerOptionTests(unittest.TestCase):
    def test_every_option_builds(self):
        for key in PLAYER_KEYS:
            agent = make_agent(key)
            self.assertEqual(agent is None, key == "human")
            self.assertIn(key, PLAYER_LABELS)

    def test_levels_are_independent(self):
        a, b = make_agent("minimax4"), make_agent("minimax2")
        self.assertIsInstance(a, MinimaxAgent)
        self.assertEqual((a.max_depth, b.max_depth), (4, 2))
        self.assertIsNot(a, b)

    def test_kinds(self):
        self.assertIsInstance(make_agent("mcts3"), MCTSAgent)
        self.assertIsInstance(make_agent("greedy"), GreedyAgent)
        self.assertIsInstance(make_agent("random"), RandomAgent)
        with self.assertRaises(ValueError):
            make_agent("minimax")  # no level -> invalid

    def test_presets(self):
        for mode in MODE_NAMES:
            for key in preset_players(mode, 3):
                self.assertIn(key, PLAYER_KEYS)
        self.assertEqual(preset_players("ai-ai", 4), ("minimax4", "minimax4"))


if __name__ == "__main__":
    unittest.main()