import chess
import gym
import gym_chess
import numpy as np
from preprocessing import fen_preprocessing as fp


class CustomEnvironment(gym.Env):
    def __init__(self, mask_moves=True):
        super(CustomEnvironment, self).__init__()
        self.__env = gym.make('ChessAlphaZero-v0')
        self.__board = chess.Board()
        self.observation_shape = (8, 8, 13)
        self.action_space = self.__env.action_space
        self.__mask_moves=mask_moves

    def render(self, mode="unicode"):
        self.__env.render(mode=mode)

    def reset(self):
        self.__env.reset()
        self.__board.reset()
        current_position = self.__board.fen()
        state = np.transpose(fp.fen_to_sparse_matrix12(current_position))
        return state

    def step(self, action):

        if self.__mask_moves:
            reward = 0
            done = False
            action = self.__env.decode(action)
            if self.__board.is_capture(action):
                if self.__board.is_en_passant(action):
                    reward += 1
                else:
                    piece = self.__board.piece_at(action.to_square)
                    if piece.piece_type == chess.PAWN:
                        reward += 1
                    elif piece.piece_type == chess.KNIGHT or piece.piece_type == chess.BISHOP:
                        reward += 3
                    elif piece.piece_type == chess.ROOK:
                        reward += 5
                    elif piece.piece_type == chess.QUEEN:
                        reward += 9

            self.__board.push(action)
            self.__env.step(self.__env.encode(action))

            if self.__board.is_game_over():
                done = True
                if self.__board.is_checkmate():
                    reward += 100

            current_position = self.__board.fen()
            state = np.transpose(fp.fen_to_sparse_matrix12(current_position))
            return state, reward, done, None

        else:

            if action not in self.__env.legal_actions:
                state=self.reset()
                return state,-1,True,None

            done=False
            action = self.__env.decode(action)
            self.__board.push(action)
            self.__env.step(self.__env.encode(action))
            if self.__board.is_game_over():
                done = True
            current_position = self.__board.fen()
            state = np.transpose(fp.fen_to_sparse_matrix12(current_position))
            return state, 10, done, None

    def get_legal_moves(self):
        return self.__env.legal_actions

    def get_ilegal_moves(self):
        valid_moves=set(self.__env.legal_actions)
        rez=[]
        for i in range(self.__env.action_space.n):
            if i not in valid_moves:
                rez.append(i)
        return rez
