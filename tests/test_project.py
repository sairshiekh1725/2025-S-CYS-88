import os
import pickle
import tempfile
import unittest

from tictactoe.agent import Qlearner
from tictactoe.game import Game, getStateKey
from tictactoe.teacher import Teacher


class AgentTests(unittest.TestCase):
    def test_q_update_terminal_and_save(self):
        agent = Qlearner(alpha=0.5, gamma=0.9, eps=0.0)
        state = "---------"
        action = (1, 1)

        agent.update(state, None, action, None, 1)

        self.assertEqual(agent.rewards, [1])
        self.assertGreater(agent.Q[action][state], 0)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "agent.pkl")
            agent.save_agent(path)
            self.assertTrue(os.path.isfile(path))
            with open(path, "rb") as file:
                loaded = pickle.load(file)
            self.assertEqual(loaded.rewards, [1])


class GameTests(unittest.TestCase):
    def test_row_win(self):
        game = Game(Qlearner(0.5, 0.9, 0.0))
        game.board = [
            ["O", "O", "O"],
            ["-", "X", "-"],
            ["X", "-", "-"],
        ]
        self.assertTrue(game.checkForWin("O"))

    def test_diagonal_win(self):
        game = Game(Qlearner(0.5, 0.9, 0.0))
        game.board = [
            ["X", "-", "O"],
            ["-", "X", "O"],
            ["-", "-", "X"],
        ]
        self.assertTrue(game.checkForWin("X"))

    def test_draw_and_state_key(self):
        board = [
            ["X", "O", "X"],
            ["X", "O", "O"],
            ["O", "X", "X"],
        ]
        game = Game(Qlearner(0.5, 0.9, 0.0))
        game.board = board
        self.assertTrue(game.checkForDraw())
        self.assertEqual(getStateKey(board), "XOXXOOOXX")


class TeacherTests(unittest.TestCase):
    def test_teacher_wins_immediately(self):
        teacher = Teacher(level=1.0)
        board = [
            ["X", "X", "-"],
            ["O", "-", "-"],
            ["O", "-", "-"],
        ]
        self.assertEqual(teacher.makeMove(board), (0, 2))


if __name__ == "__main__":
    unittest.main()
