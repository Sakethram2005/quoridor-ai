import random
import unittest

from quoridor.game import State, GOAL_ROW, N
from quoridor.pathfinding import astar, bfs_distance
from quoridor.agent import (MinimaxAgent, GreedyAgent, RACE_SCORE, WIN_SCORE,
                            improved_evaluate, candidate_walls)
from quoridor.mcts import MCTSAgent


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


def random_board(seed, n_walls=14):
    rng = random.Random(seed)
    s = State()
    for _ in range(n_walls):
        nxt = s.try_wall(rng.randrange(8), rng.randrange(8), rng.choice("HV"), player=rng.randrange(2))
        if nxt is not None:
            s = nxt
    return s


class DistanceFieldTests(unittest.TestCase):
    def test_field_matches_astar(self):
        for seed in range(25):
            s = random_board(seed)
            for p in (0, 1):
                field = s.dist_field(p)
                self.assertEqual(field[s.pawns[p]], s.dist(p))

    def test_options_counts_optimal_first_steps(self):
        s = State()
        self.assertEqual(s.options(0), 1)                       # straight ahead only
        s = State(pawns=((4, 4), (0, 4)))
        self.assertEqual(s.options(0), 1)
        s = State(pawns=((4, 4), (0, 4)), walls={(3, 4, "H")})  # wall ahead: detour
        self.assertEqual(s.dist(0), 5)                          # wall spans cols 4-5: go left
        self.assertEqual(s.options(0), 1)


class EvaluationTests(unittest.TestCase):
    def test_race_rule(self):
        # opponent has no walls and I am clearly ahead -> decided
        s = State(pawns=((3, 4), (2, 4)), walls_left=(5, 0), turn=0)
        self.assertGreaterEqual(improved_evaluate(s, 0), RACE_SCORE)
        # I have no walls and the opponent is ahead -> lost
        s = State(pawns=((8, 4), (7, 4)), walls_left=(0, 5), turn=0)
        self.assertLessEqual(improved_evaluate(s, 0), -RACE_SCORE + 100)

    def test_terminal_scores(self):
        s = State(pawns=((0, 3), (5, 5)))
        self.assertEqual(improved_evaluate(s, 0), WIN_SCORE)
        self.assertEqual(improved_evaluate(s, 1), -WIN_SCORE)

    def test_candidate_walls_all_fit(self):
        s = random_board(3, 8)
        for w in candidate_walls(s, 1):
            self.assertTrue(s.wall_fits(*w))


class MinimaxVariantTests(unittest.TestCase):
    def test_transposition_table_does_not_change_value(self):
        for seed in range(4):
            s = random_board(seed, 10)
            vals = []
            for tt in (True, False):
                a = MinimaxAgent(3, 30, use_tt=tt, weights={"flex": 0.0})
                a.choose_move(s)
                vals.append(a.last_stats["score"])
            self.assertAlmostEqual(vals[0], vals[1], places=6, msg=f"seed {seed}")

    def test_basic_mode_still_plays_legally(self):
        s = State()
        agents = [MinimaxAgent(2, 1, improved=False), MinimaxAgent(2, 1, improved=True, seed=1)]
        for _ in range(30):
            if s.winner() is not None:
                break
            mv = agents[s.turn].choose_move(s)
            self.assertTrue(s.is_legal(mv), mv)
            s = s.apply(mv)


class MCTSTests(unittest.TestCase):
    def test_takes_immediate_win(self):
        s = State(pawns=((1, 4), (6, 0)), turn=0)
        self.assertEqual(MCTSAgent(50, seed=1).choose_move(s), ("move", 0, 4))

    def test_blocks_opponent_about_to_win(self):
        s = State(pawns=((8, 4), (7, 4)), turn=0)
        self.assertEqual(MCTSAgent(300, seed=1).choose_move(s)[0], "wall")

    def test_moves_are_always_legal(self):
        s = State()
        agents = [MCTSAgent(60, seed=2), GreedyAgent()]
        for _ in range(40):
            if s.winner() is not None:
                break
            mv = agents[s.turn].choose_move(s)
            self.assertTrue(s.is_legal(mv), mv)
            s = s.apply(mv)

    def test_mcts_beats_greedy(self):
        for mc_player in (0, 1):
            agents = {mc_player: MCTSAgent(300, seed=3), 1 - mc_player: GreedyAgent()}
            s = State()
            for _ in range(200):
                if s.winner() is not None:
                    break
                s = s.apply(agents[s.turn].choose_move(s))
            self.assertEqual(s.winner(), mc_player)


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
