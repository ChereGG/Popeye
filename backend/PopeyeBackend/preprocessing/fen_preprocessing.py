import chess
import chess.engine
import chess.pgn


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


def save_train_data(fen_dict, filePath):
    with open(filePath, 'w') as file:
        for fen in fen_dict:
            file.write(str(fen) + ":" + str(fen_dict[fen]) + "\n")


def create_labels(pgn_file):
    pgn = open(pgn_file)
    fen_dict = dict()
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish.exe")
    i=1
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(depth=15))
            fen_dict[board.fen()] = info['score'].white()
        print("Game: "+str(i)+" ended!")
    engine.quit()
    return fen_dict


def main():
    fen_dict = create_labels("Carlsen.pgn")
    save_train_data(fen_dict, "trainData")
