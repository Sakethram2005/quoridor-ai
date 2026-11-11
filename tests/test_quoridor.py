import random
import unittest

from quoridor.game import State, GOAL_ROW, N
from quoridor.pathfinding import astar, bfs_distance
from quoridor.agent import MinimaxAgent, GreedyAgent


class RulesTests(unittest.TestCase):
    def test_start_position(self):
        s = State()
        self.assertEqual(s.pawns, ((8, 4), (0, 4)))
        self.assertEqual(sorted(s.pawn_moves(0)), [(7, 4), (8, 3), (8, 5)])
        self.assertEqual(s.dist(0), 8)
        self.assertEqual(s.dist(1), 8)
        self.assertEqual(s.walls_left, (10, 10))

    def test_wall_blocks_pawn(self):
        s = State().try_wall(7, 4, "H", player=0)       # blocks (7,4)-(8,4) and (7,5)-(8,5)
        self.assertEqual(sorted(s.pawn_moves(0)), [(8, 3), (8, 5)])
        self.assertEqual(s.walls_left[0], 9)

    def test_wall_overlap_and_cross_rejected(self):
        s = State().try_wall(3, 3, "H", player=0)
        self.assertIsNone(s.try_wall(3, 3, "H", player=1))   # same place
        self.assertIsNone(s.try_wall(3, 2, "H", player=1))   # overlaps left
        self.assertIsNone(s.try_wall(3, 4, "H", player=1))   # overlaps right
        self.assertIsNone(s.try_wall(3, 3, "V", player=1))   # crosses
        self.assertIsNotNone(s.try_wall(3, 5, "H", player=1))
        self.assertIsNone(s.try_wall(8, 0, "H", player=1))   # out of bounds

    def test_no_walls_left(self):
        s = State(walls_left=(0, 10))
        self.assertIsNone(s.try_wall(3, 3, "H", player=0))

    def test_wall_may_not_seal_a_player_in(self):
        s = State()
        for r, c, o in [(3, 0, "H"), (3, 2, "H"), (3, 4, "H"), (3, 6, "H"), (3, 7, "V")]:
            s = s.try_wall(r, c, o, player=0)
            self.assertIsNotNone(s, (r, c, o))
        self.assertIsNone(s.try_wall(4, 7, "H", player=1))   # would cut off every route

    def test_straight_jump(self):
        s = State(pawns=((4, 4), (3, 4)), turn=0)
        self.assertIn((2, 4), s.pawn_moves(0))
        self.assertNotIn((3, 4), s.pawn_moves(0))

    def test_diagonal_jump_when_wall_behind_opponent(self):
        s = State(pawns=((4, 4), (3, 4)), walls={(2, 4, "H")}, turn=0)
        moves = s.pawn_moves(0)
        self.assertNotIn((2, 4), moves)
        self.assertIn((3, 3), moves)
        self.assertIn((3, 5), moves)

    def test_winner(self):
        self.assertEqual(State(pawns=((0, 3), (5, 5))).winner(), 0)
        self.assertEqual(State(pawns=((4, 3), (8, 5))).winner(), 1)
        self.assertIsNone(State().winner())

    def test_apply_rejects_illegal(self):
        with self.assertRaises(ValueError):
            State().apply(("move", 5, 4))
        with self.assertRaises(ValueError):
            State().apply(("wall", 0, 0, "X"))


class PathTests(unittest.TestCase):
    def test_astar_matches_bfs_on_random_boards(self):
        rng = random.Random(1)
        for _ in range(40):
            s = State()
            for _ in range(rng.randint(0, 14)):
                nxt = s.try_wall(rng.randrange(8), rng.randrange(8), rng.choice("HV"), player=rng.randrange(2))
                if nxt is not None:
                    s = nxt
            for p in (0, 1):
                path = astar(s, s.pawns[p], GOAL_ROW[p])
                self.assertEqual(len(path) - 1, bfs_distance(s, s.pawns[p], GOAL_ROW[p]))
                for a, b in zip(path, path[1:]):
                    self.assertIn(b, list(s.neighbors(a)))


class AgentTests(unittest.TestCase):
    def test_takes_immediate_win(self):
        s = State(pawns=((1, 4), (6, 0)), turn=0)
        self.assertEqual(MinimaxAgent(2, 5).choose_move(s), ("move", 0, 4))

    def test_blocks_opponent_about_to_win(self):
        # Red (player 1) is one step from row 8 and Blue is far away: the only
        # way to survive is a wall that lengthens Red's path.
        s = State(pawns=((8, 4), (7, 4)), turn=0)
        mv = MinimaxAgent(3, 5).choose_move(s)
        self.assertEqual(mv[0], "wall")

    def test_moves_are_always_legal(self):
        s = State()
        agents = [MinimaxAgent(2, 1), GreedyAgent()]
        for _ in range(40):
            if s.winner() is not None:
                break
            mv = agents[s.turn].choose_move(s)
            self.assertTrue(s.is_legal(mv), mv)
            s = s.apply(mv)

    def test_minimax_beats_greedy(self):
        for mm_player in (0, 1):
            agents = {mm_player: MinimaxAgent(2, 2), 1 - mm_player: GreedyAgent()}
            s = State()
            for _ in range(200):
                if s.winner() is not None:
                    break
                s = s.apply(agents[s.turn].choose_move(s))
            self.assertEqual(s.winner(), mm_player)


if __name__ == "__main__":
    unittest.main()