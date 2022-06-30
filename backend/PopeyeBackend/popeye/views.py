import gym
import gym_chess
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from preprocessing import fen_preprocessing as fp
import chess
import tensorflow as tf
import numpy as np
import json
from models.supervised.auto_encoder import AutoEncoderAccuracy
tf.config.set_visible_devices([], 'GPU')

model_reinforcement = tf.keras.models.load_model("./models/reinforcement/saved-models/dqn-agent-masked-moves-resnet-alpha0/model")
env = gym.make('ChessAlphaZero-v0')
model_superised = tf.keras.models.load_model("./models/supervised/saved-models/conv_model_classificator/")
auto_enc = tf.keras.models.load_model("./models/supervised/saved-models/auto_encoder_unet_model_8/",custom_objects={"AutoEncoderAccuracy": AutoEncoderAccuracy})
encoder = auto_enc.layers[:10]
encoder = tf.keras.models.Sequential(encoder)

@api_view(['POST'])
def send_move_reinforcement(request):
    if request.method == 'POST':
        body = json.loads(request.body.decode("utf-8"))
        move = body["move"]
        step = None
        if move == "start":
            step = env.reset()
        elif move != "start":
            step, _, _, _, = env.step(env.encode(chess.Move.from_uci(move)))
        step = np.expand_dims(step, axis=0)
        output = model_reinforcement(step, training=False)[0]
        all_moves = np.array(list(range(output.shape[0])))
        all_legal_moves = env.legal_actions
        illegal_moves = [move for move in all_moves if move not in all_legal_moves]
        illegal_moves = np.expand_dims(illegal_moves, axis=-1)
        encoded_legal_moves = tf.tensor_scatter_nd_update(output, illegal_moves, [np.NINF] * len(illegal_moves))
        move = tf.argmax(encoded_legal_moves).numpy()
        move = env.decode(move)
        env.step(env.encode(move))
        return JsonResponse({'move': str(move)}, status=status.HTTP_200_OK)


@api_view(['POST'])
def send_move_supervised(request):
    if request.method == 'POST':
        fen = request.data['fen']
        board = chess.Board(fen=fen)
        next_moves = board.legal_moves
        for move in next_moves:
            board.push(move)
            fen = board.fen()
            matrix = np.array(fp.fen_to_sparse_matrix12(fen)).transpose()
            input = encoder(np.array([matrix.flatten()]), training=False)[0]
            logits = model_superised(np.array([input]), training=False)[0]
            label = tf.argmax(logits)
            board.pop()
            if label == 2:
                return JsonResponse({'move': str(move)}, status=status.HTTP_200_OK)
        return JsonResponse({'move': str(np.random.choice(list(next_moves)))}, status=status.HTTP_200_OK)

@api_view(['POST'])
def undo_reinforcement(request):
    if request.method == 'POST':
        move_history=request.data['move_history']
        env.reset()
        for move in move_history:
            env.step(env.encode(chess.Move.from_uci(move)))
        return JsonResponse({'move': 'OK'}, status=status.HTTP_200_OK)