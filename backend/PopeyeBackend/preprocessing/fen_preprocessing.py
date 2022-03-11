import chess
import chess.engine
import chess.pgn
import os
from decouple import config
import numpy as np


def fen_to_matrix(fen):
    piece_values = {"P": 1, "R": 5, "N": 3, "B": 3, "K": 100, "Q": 9, "p": -1, "r": -5, "n": -3, "b": -3, "k": -100,
                    "q": -9}
    matrix = [[0 for _ in range(8)] for _ in range(8)]
    records = fen.split()
    board = records[0]
    position_rows = board.split("/")
    contor = 0
    for position_row in position_rows:
        for potential_piece in position_row:
            if potential_piece.isalpha():
                matrix[int(contor / 8)][contor % 8] = piece_values[potential_piece] / piece_values["K"]
                contor += 1
            else:
                contor += int(potential_piece)
    return matrix


def fen_to_sparse_matrix6(fen):
    piece_map = {
        "P": [[0 for _ in range(8)] for _ in range(8)],
        "R": [[0 for _ in range(8)] for _ in range(8)],
        "N": [[0 for _ in range(8)] for _ in range(8)],
        "B": [[0 for _ in range(8)] for _ in range(8)],
        "K": [[0 for _ in range(8)] for _ in range(8)],
        "Q": [[0 for _ in range(8)] for _ in range(8)],
    }
    records = fen.split()
    board = records[0]
    position_rows = board.split("/")
    contor = 0
    for position_row in position_rows:
        for potential_piece in position_row:
            if potential_piece.isalpha():
                if potential_piece.isupper():
                    piece_map[potential_piece][int(contor / 8)][contor % 8] = 1
                else:
                    piece_map[potential_piece.upper()][int(contor / 8)][contor % 8] = -1
                contor += 1
            else:
                contor += int(potential_piece)
    return [piece_map["P"], piece_map["R"], piece_map["N"],
            piece_map["B"], piece_map["K"], piece_map["Q"]]


def fen_to_sparse_matrix12(fen):
    piece_map = {
        "P": [[0 for _ in range(8)] for _ in range(8)],
        "R": [[0 for _ in range(8)] for _ in range(8)],
        "N": [[0 for _ in range(8)] for _ in range(8)],
        "B": [[0 for _ in range(8)] for _ in range(8)],
        "K": [[0 for _ in range(8)] for _ in range(8)],
        "Q": [[0 for _ in range(8)] for _ in range(8)],
        "p": [[0 for _ in range(8)] for _ in range(8)],
        "r": [[0 for _ in range(8)] for _ in range(8)],
        "n": [[0 for _ in range(8)] for _ in range(8)],
        "b": [[0 for _ in range(8)] for _ in range(8)],
        "k": [[0 for _ in range(8)] for _ in range(8)],
        "q": [[0 for _ in range(8)] for _ in range(8)],
    }
    records = fen.split()
    board = records[0]
    position_rows = board.split("/")
    contor = 0
    for position_row in position_rows:
        for potential_piece in position_row:
            if potential_piece.isalpha():
                piece_map[potential_piece][int(contor / 8)][contor % 8] = 1
                contor += 1
            else:
                contor += int(potential_piece)
    return [piece_map["P"], piece_map["R"], piece_map["N"],
            piece_map["B"], piece_map["K"], piece_map["Q"], piece_map["p"], piece_map["r"], piece_map["n"],
            piece_map["b"], piece_map["k"], piece_map["q"]]

def getLines(inputFilePath):
    with open(inputFilePath, 'r') as file:
        lines=file.readlines()
    return lines

def getTrainDataDenseMatrix(inputFilePath):
    X_train = np.empty(shape=(0, 8, 8))
    Y_train = np.empty(shape=(0))
    with open(inputFilePath, 'r') as file:
        for line in file:
            line_elems = line.split(":")
            fen, evaluation = line_elems[0], line_elems[1]
            if "#" in evaluation:
                continue
            X_train = np.append(X_train, np.array([fen_to_matrix(fen)]), axis=0)
            Y_train = np.append(Y_train, np.array([int(evaluation)]), axis=0)
    return X_train, Y_train


def getTrainDataSparseMatrix6(inputFilePath):
    X_train = np.empty(shape=(0, 6, 8, 8))
    Y_train = np.empty(shape=(0))
    with open(inputFilePath, 'r') as file:
        for line in file:
            line_elems = line.split(":")
            fen, evaluation = line_elems[0], line_elems[1]
            if "#" in evaluation:
                continue
            X_train = np.append(X_train, np.array([fen_to_sparse_matrix6(fen)]), axis=0)
            Y_train = np.append(Y_train, np.array([int(evaluation)]), axis=0)
    return X_train, Y_train


def getTrainDataSparseMatrix12Classification(lines):
    X_train = np.zeros(shape=(len(lines), 12, 8, 8))
    Y_train = np.zeros(shape=(len(lines), 1))
    i = 0
    for idx, line in enumerate(lines):
        line = line.strip()
        line_elems = line.split(":")
        fen, evaluation = line_elems[0], line_elems[1]
        if "#" in evaluation:
            continue
        X_train[idx] = np.array(fen_to_sparse_matrix12(fen))
        if int(evaluation) < -100:
            Y_train[idx] = np.array([0])
        elif -100 < int(evaluation) < 100:
            Y_train[idx] = np.array([1])
        else:
            Y_train[idx] = np.array([2])
        i += 1
        if i % 10000 == 0:
            print("Gata inca 10000 suntem la : " + str(i))
    return X_train, Y_train


def save_train_data(fen_dict, batchPath):
    with open(batchPath + "/trainData", 'a') as file:
        for fen in fen_dict:
            file.write(str(fen) + ":" + str(fen_dict[fen]) + "\n")


def save_checkpoint(batchPath, idxPgn, idxGame):
    with open(batchPath + "/checkpoint.txt", "w") as file:
        file.write(str(idxPgn) + ":" + str(idxGame))


def read_checkpoint(batchPath):
    with open(batchPath + "/checkpoint.txt", "r") as file:
        idxGame, idxCheckpont = file.read().strip().split(":")
    return int(idxGame), int(idxCheckpont)


def create_labels(batchPath):
    idxGame, idxPgn = read_checkpoint(batchPath)

    engine = chess.engine.SimpleEngine.popen_uci("../../stockfish/stockfish.exe")
    for idx, pgn_file in enumerate(os.listdir(batchPath)):
        if ".txt" in pgn_file or idx < idxGame:
            continue
        pgn_file = batchPath + "/" + pgn_file
        pgn = open(pgn_file)
        fen_dict = dict()
        i = 1
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                if i % 10 != 0:
                    print("Saving...")
                    save_train_data(fen_dict, batchPath)
                    save_checkpoint(batchPath, idx, i)
                    fen_dict.clear()
                break
            if idx == idxGame and i <= idxPgn:
                i += 1
                continue
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                info = engine.analyse(board, chess.engine.Limit(depth=15))
                fen_dict[board.fen()] = info['score'].white()
            print("Game: " + str(i) + " from " + pgn_file.split("/")[-1] + " ended!")
            if i % 10 == 0:
                print("Saving...")
                save_train_data(fen_dict, batchPath)
                save_checkpoint(batchPath, idx, i)
                fen_dict.clear()
            i += 1
        pgn.close()
    engine.quit()


def remove_duplicates(input_file, output_file):
    fens = set()
    evaluations = []
    i = 0
    with open(input_file, 'r') as file:
        for line in file:
            line_elems = line.split(":")
            fen, evaluation = line_elems[0], line_elems[1]
            if "#" in evaluation:
                continue
            if fen not in fens:
                fens.add(fen)
                evaluations.append(evaluation)
            i += 1
            if i % 10000 == 0:
                print("Gata inca 10000 suntem la : " + str(i))
    with open(output_file, 'w') as file:
        for fen, eval in zip(fens, evaluations):
            file.write(fen + ":" + str(eval))


def main():
    remove_duplicates("../trainData","../theData")


if __name__ == '__main__':
    main()
