from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from preprocessing import fen_preprocessing as fp
from models import conv_model as cm
import chess

model = cm.get_loaded_model()


@api_view(['POST'])
def next_move(request):
    if request.method == 'POST':
        fen = request.data['fen']
        print(fen)

        board = chess.Board(fen=fen)
        next_moves = board.legal_moves
        max_eval, best_move = None, None
        for move in next_moves:
            board.push(move)
            fen = board.fen()
            color = board.turn
            matrix = fp.fen_to_matrix(fen)
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
