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
import json

tf.config.set_visible_devices([], 'GPU')
model = None
env = None

if config("DQN-AGENT") == "True":
    model = tf.keras.models.load_model("./models/reinforcement/dqn-agent-masked-moves-resnet-alpha0/model")
    env = gym.make('ChessAlphaZero-v0')
else:
    model = tf.keras.models.load_model("./models/supervised/conv_model/")


@api_view(['POST'])
def send_move(request):
    if request.method == 'POST':
        if config("DQN-AGENT") == "True":
            body = json.loads(request.body.decode("utf-8"))
            move = body["move"]
            step = None
            if move == "start":
                step = env.reset()
            elif move != "start":
                step, _, _, _, = env.step(env.encode(chess.Move.from_uci(move)))
            step = np.expand_dims(step, axis=0)
            output = model(step, training=False)[0]
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
