import gym
import gym_chess
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from preprocessing import fen_preprocessing as fp
import chess
import tensorflow as tf
from decouple import config
import numpy as np

model = None
env = None

if config("DQN-AGENT") == "True":
    model = tf.keras.models.load_model("./models/reinforcement/dqn-agent-masked-moves/")
    env = gym.make('ChessAlphaZero-v0')
    env.reset()
else:
    model = tf.keras.models.load_model("./models/supervised/conv_model/")


@api_view(['POST'])
def next_move(request):
    if request.method == 'POST':
        if config("DQN-AGENT") == "True":
            fen = request.data['fen']
            encoded_board = np.transpose(fp.fen_to_sparse_matrix13(fen))
            encoded_board = np.expand_dims(encoded_board, axis=0)
            output = model(encoded_board, training=False)[0]
            all_moves = np.array(list(range(output.shape[0])))
            all_legal_moves = env.legal_actions
            illegal_moves = [move for move in all_moves if move not in all_legal_moves]
            illegal_moves = np.expand_dims(illegal_moves, axis=-1)
            encoded_legal_moves = tf.tensor_scatter_nd_update(output, illegal_moves, [np.NINF] * len(illegal_moves))
            move = tf.argmax(encoded_legal_moves).numpy()
            move = env.decode(move)
            env.step(env.encode(move))
            return JsonResponse({'move': str(move)}, status=status.HTTP_200_OK)
        else:
            fen = request.data['fen']
            board = chess.Board(fen=fen)
            next_moves = board.legal_moves
            max_eval, best_move = None, None
            for move in next_moves:
                board.push(move)
                fen = board.fen()
                color = board.turn
                matrix = fp.fen_to_sparse_matrix6(fen)
                evalutaion = model.predict([matrix])[0]
                if max_eval is None:
                    max_eval = evalutaion
                    best_move = move
                elif color == chess.WHITE and max_eval < evalutaion:
                    max_eval = evalutaion
                    best_move = move
                elif color == chess.BLACK and max_eval > evalutaion:
                    max_eval = evalutaion
                    best_move = move
                board.pop()

            return JsonResponse({'move': str(best_move)}, status=status.HTTP_200_OK)

@api_view(['POST'])
def send_move(request):
    if request.method == 'POST':
        move = request.data['move']
        move = chess.Move.from_uci(move)
        env.step(env.encode(move))
        return JsonResponse({'message': 'Ok!'}, status=status.HTTP_200_OK)