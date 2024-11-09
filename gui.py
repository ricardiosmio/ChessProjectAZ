import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk
from tkinter import simpledialog
import chess
from PIL import Image, ImageTk
from chess_engine import AlphaZeroAgent, encode_board, alphazero_model  # Import the engine
import logging

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class ChessGUI:
    LIGHT_SQUARE_COLOR = "#F0D9B5"
    DARK_SQUARE_COLOR = "#B58863"

    def __init__(self, root):
        self.root = root
        self.root.title("Chess GUI")

        self.board = chess.Board()
        self.board_frame = tk.Frame(root)
        self.board_frame.pack(side=tk.LEFT)

        self.board_canvas = tk.Canvas(self.board_frame, width=480, height=480)
        self.board_canvas.grid(row=1, column=1, rowspan=8, columnspan=8)

        self.add_labels()

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.RIGHT)

        self.white_button = tk.Button(self.control_frame, text="Play as White", command=self.play_white)
        self.white_button.pack(side=tk.TOP)

        self.black_button = tk.Button(self.control_frame, text="Play as Black", command=self.play_black)
        self.black_button.pack(side=tk.TOP)

        # Add the new game button
        self.new_game_button = tk.Button(self.control_frame, text="New game", command=self.reset_game)
        self.new_game_button.pack(side=tk.TOP)

        # Add the undo button
        self.undo_button = tk.Button(self.control_frame, text="Undo Move", command=self.undo_move)
        self.undo_button.pack(side=tk.TOP)

        # Add the move suggestion button
        self.suggest_button = tk.Button(self.control_frame, text="Suggest Move", command=self.suggest_move)
        self.suggest_button.pack(side=tk.TOP)

        # Add the play multiple games button
        self.multi_game_button = tk.Button(self.control_frame, text="Play Multiple Games", command=self.play_multiple_games)
        self.multi_game_button.pack(side=tk.TOP)

        self.move_history = tk.Text(self.control_frame, width=30, height=20, state=tk.DISABLED)
        self.move_history.pack(side=tk.TOP)

        self.status_label = tk.Label(self.control_frame, text="", font=("Arial", 14))
        self.status_label.pack(side=tk.TOP)

        self.piece_images = self.load_piece_images()
        self.selected_square = None
        self.is_white_player = True  # By default, the player is white
        self.agent = AlphaZeroAgent(alphazero_model((8, 8, 12)))  # Initialize the agent
        self.highlight_last_move_from = None
        self.highlight_last_move_to = None
        self.update_board()

        self.board_canvas.bind("<Button-1>", self.on_square_click)

    def undo_move(self):
        if self.board.move_stack:
            self.board.pop()
            self.update_board()
            self.update_move_history()
            self.status_label.config(text="")

    def suggest_move(self):
        suggested_move = self.agent.act(self.board)  # Use the chess engine to suggest a move
        self.status_label.config(text=f"Suggested move: {suggested_move}")

        # Highlight the suggested move
        self.update_board(highlight_from=suggested_move.from_square, highlight_to=suggested_move.to_square)

    def reset_game(self):
        self.board = chess.Board()
        self.highlight_last_move_from = None
        self.highlight_last_move_to = None
        self.update_board()
        self.update_move_history()
        self.status_label.config(text="")
        if not self.is_white_player:
            self.engine_move()

    def add_labels(self):
        self.row_labels = []
        self.col_labels = []
        for i in range(8):
            col_label_top = tk.Label(self.board_frame, text=chr(65 + i), font=("Arial", 12))
            col_label_top.grid(row=0, column=i+1)
            self.col_labels.append(col_label_top)

            col_label_bottom = tk.Label(self.board_frame, text=chr(65 + i), font=("Arial", 12))
            col_label_bottom.grid(row=9, column=i+1)
            self.col_labels.append(col_label_bottom)

            row_label_left = tk.Label(self.board_frame, text=str(8 - i), font=("Arial", 12))
            row_label_left.grid(row=i+1, column=0)
            self.row_labels.append(row_label_left)

            row_label_right = tk.Label(self.board_frame, text=str(8 - i), font=("Arial", 12))
            row_label_right.grid(row=i+1, column=9)
            self.row_labels.append(row_label_right)

    def update_labels(self):
        for i, label in enumerate(self.col_labels):
            label.config(text=chr(65 + (i // 2) if self.is_white_player else 65 + (7 - i // 2)))
        for i, label in enumerate(self.row_labels):
            label.config(text=str(8 - (i // 2) if self.is_white_player else 1 + (i // 2)))

    def load_piece_images(self):
        pieces = {
            'r': 'DR', 'n': 'DN', 'b': 'DB', 'q': 'DQ', 'k': 'DK', 'p': 'DP',
            'R': 'WR', 'N': 'WN', 'B': 'WB', 'Q': 'WQ', 'K': 'WK', 'P': 'WP'
        }
        piece_images = {}
        for piece, filename in pieces.items():
            image = Image.open(f"images/{filename}.png")
            piece_images[piece] = ImageTk.PhotoImage(image.resize((60, 60)))

        # Load the transparent yellow highlight image
        highlight_image = Image.open("images/yellow_transparent.png")
        self.highlight_image = ImageTk.PhotoImage(highlight_image.resize((60, 60)))
        
        return piece_images

    def play_white(self):
        self.board = chess.Board()
        self.is_white_player = True
        self.highlight_last_move_from = None
        self.highlight_last_move_to = None
        self.update_board()
        self.update_labels()
        self.update_move_history()
        self.status_label.config(text="")

    def play_black(self):
        self.board = chess.Board()
        self.is_white_player = False
        self.highlight_last_move_from = None
        self.highlight_last_move_to = None
        self.update_board()
        self.update_labels()
        self.update_move_history()
        self.status_label.config(text="")
        self.engine_move()  # Engine makes the first move

    def update_board(self, highlight_from=None, highlight_to=None):
        self.board_canvas.delete("all")
        for square in chess.SQUARES:
            x = (square % 8) * 60
            y = (7 - square // 8) * 60 if self.is_white_player else (square // 8) * 60
            color = self.LIGHT_SQUARE_COLOR if (square + square // 8) % 2 == 0 else self.DARK_SQUARE_COLOR

            # Highlight the selected piece and destination square
            if square == highlight_from or square == highlight_to:
                color = "#FFFF00"  # Highlight color

            self.board_canvas.create_rectangle(x, y, x + 60, y + 60, fill=color)

        for square in chess.SQUARES:
            x = (square % 8) * 60
            y = (7 - square // 8) * 60 if self.is_white_player else (square // 8) * 60

            # Highlight the last move with the transparent image
            if square == self.highlight_last_move_from or square == self.highlight_last_move_to:
                self.board_canvas.create_image(x, y, image=self.highlight_image, anchor=tk.NW)

            piece = self.board.piece_at(square)
            if piece is not None:
                piece_image = self.piece_images.get(piece.symbol())
                if piece_image:
                    self.board_canvas.create_image(x, y, image=piece_image, anchor=tk.NW)

    def on_square_click(self, event):
        col = event.x // 60
        row = 7 - (event.y // 60) if self.is_white_player else event.y // 60
        square = chess.square(col, row)

        if self.selected_square is None:
            self.selected_square = square
            self.highlight_moves(square)
        else:
            piece = self.board.piece_at(self.selected_square)  # Retrieve the piece at the selected square
            if piece and piece.piece_type == chess.PAWN and (
                (piece.color == chess.WHITE and chess.square_rank(square) == 7) or 
                (piece.color == chess.BLACK and chess.square_rank(square) == 0)
            ):
                self.promote_pawn(self.selected_square, square)
            else:
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.highlight_last_move_from = self.selected_square
                    self.highlight_last_move_to = square
                    self.update_board()
                    self.update_move_history()
                    self.check_game_status()
                    self.engine_move()
                else:
                    self.status_label.config(text="Illegal move, resetting selection")

            self.selected_square = None
            self.update_board()

    def promote_pawn(self, from_square, to_square):
        x, y = self.root.winfo_pointerxy()

        promotion_window = tk.Toplevel(self.root)
        promotion_window.title("Promote Pawn")
        promotion_window.geometry(f"300x150+{x}+{y}")

        pawn_color = 'W' if self.board.piece_at(from_square).color == chess.WHITE else 'D'
        label = tk.Label(promotion_window, text="Promote pawn to:")
        label.pack()

        def set_promotion(piece_type):
            move = chess.Move(from_square, to_square, promotion=piece_type)

            if move in self.board.legal_moves:
                self.board.push(move)
                promotion_window.destroy()
                self.highlight_last_move_from = from_square
                self.highlight_last_move_to = to_square
                self.update_board()
                self.update_move_history()
                self.check_game_status()
                self.engine_move()
            else:
                promotion_window.destroy()

        tk.Button(promotion_window, text="Queen", command=lambda: set_promotion(chess.QUEEN)).pack(side=tk.LEFT)
        tk.Button(promotion_window, text="Rook", command=lambda: set_promotion(chess.ROOK)).pack(side=tk.LEFT)
        tk.Button(promotion_window, text="Bishop", command=lambda: set_promotion(chess.BISHOP)).pack(side=tk.LEFT)
        tk.Button(promotion_window, text="Knight", command=lambda: set_promotion(chess.KNIGHT)).pack(side=tk.LEFT)

    def highlight_moves(self, square):
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            if move.from_square == square:
                self.board_canvas.create_rectangle(
                    (move.to_square % 8) * 60, (7 - move.to_square // 8) * 60 if self.is_white_player else (move.to_square // 8) * 60,
                    (move.to_square % 8 + 1) * 60, (8 - move.to_square // 8) * 60 if self.is_white_player else (move.to_square // 8 + 1) * 60,
                    outline="yellow", width=3
                )

    def update_move_history(self):
        self.move_history.config(state=tk.NORMAL)
        self.move_history.delete('1.0', tk.END)
        move_list = list(self.board.move_stack)
        for i, move in enumerate(move_list):
            if i % 2 == 0:
                self.move_history.insert(tk.END, f"{i//2 + 1}. {move} ")
            else:
                self.move_history.insert(tk.END, f"{move}\n")
        self.move_history.config(state=tk.DISABLED)

    def check_game_status(self):
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            self.status_label.config(text=f"Checkmate! {winner} wins.")
        elif self.board.is_stalemate():
            self.status_label.config(text="Stalemate!")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Draw by insufficient material.")
        elif self.board.is_seventyfive_moves():
            self.status_label.config(text="Draw by 75-move rule.")
        elif self.board.is_variant_draw():
            self.status_label.config(text="Draw!")

    def engine_move(self):
        if not self.board.is_game_over():
            move = self.agent.act(self.board)
            if move is not None:
                self.board.push(move)
                self.highlight_last_move_from = move.from_square
                self.highlight_last_move_to = move.to_square
                self.update_board()
                self.update_move_history()
                self.check_game_status()
            else:
                self.status_label.config(text="Engine failed to produce a move.")
        else:
            self.check_game_status()

    def play_multiple_games(self):
        num_games = simpledialog.askinteger("Input", "How many games do you want to play?")
        if num_games:
            self.agent.train_agent(num_games)

def main():
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
