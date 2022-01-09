import chess
import chess.engine
import chess.pgn
import os


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


def save_train_data(fen_dict, filePath):
    with open(filePath, 'a') as file:
        for fen in fen_dict:
            file.write(str(fen) + ":" + str(fen_dict[fen]) + "\n")


def save_checkpoint(idxPgn, idxGame):
    with open("../games/checkpoint.txt", "w") as file:
        file.write(str(idxPgn) + ":" + str(idxGame))


def readCheckpoint():
    with open("../games/checkpoint.txt", "r") as file:
        idxGame, idxCheckpont = file.read().strip().split(":")
    return int(idxGame), int(idxCheckpont)


def create_labels(pgn_folder):
    idxGame, idxPgn = readCheckpoint()

    engine = chess.engine.SimpleEngine.popen_uci("../../stockfish/stockfish.exe")
    for idx, pgn_file in enumerate(os.listdir(pgn_folder)):
        if ".txt" in pgn_file or idx < idxGame:
            continue
        pgn_file = pgn_folder + "/" + pgn_file
        pgn = open(pgn_file)
        fen_dict = dict()
        i = 1
        while True:
            if idx == idxGame and i <= idxPgn:
                i += 1
                continue
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                info = engine.analyse(board, chess.engine.Limit(depth=15))
                fen_dict[board.fen()] = info['score'].white()
            print("Game: " + str(i) + " from " + pgn_file.split("/")[-1] + " ended!")
            if i % 10 == 0:
                print("Saving...")
                save_train_data(fen_dict, '../trainData')
                save_checkpoint(idx, i)
            i += 1
        engine.quit()
        pgn.close()


def main():
    create_labels("../games/")


if __name__ == '__main__':
    main()
