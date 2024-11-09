import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import os
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Parameters
GAMMA = 0.99
ALPHA = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPISODES = 1000  # Number of games to play for training

def resnet_block(input_tensor, filters, kernel_size=3):
    x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def alphazero_model(input_shape, num_res_blocks=19):
    inputs = Input(shape=input_shape)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(inputs)
    
    for _ in range(num_res_blocks):
        x = resnet_block(x, 256)
    
    # Policy head
    policy = Conv2D(2, (1, 1), padding='same', activation='relu')(x)
    policy = Flatten()(policy)
    policy = Dense(128, activation='relu')(policy)
    policy = Dense(4672, activation='softmax')(policy)  # 4672 possible moves in chess
    
    # Value head
    value = Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    value = Flatten()(value)
    value = Dense(256, activation='relu')(value)
    value = Dense(1, activation='tanh')(value)
    
    model = Model(inputs=inputs, outputs=[policy, value])
    model.compile(optimizer=Adam(learning_rate=ALPHA), loss=['categorical_crossentropy', 'mse'], metrics=['accuracy'])
    
    return model

def encode_board(board):
    encoded = np.zeros((8, 8, 12), dtype=int)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.symbol()
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            encoded[rank][file][piece_map[piece_type]] = 1
    return encoded

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self._node = {}

    def _playout(self, state):
        # Implement the playout logic
        pass

    def get_move(self, state):
        for _ in range(self.n_playout):
            self._playout(state)
        # Return the most visited move
        pass

class AlphaZeroAgent:
    def __init__(self, model):
        self.model = model
        self.mcts = MCTS(self.policy_value_fn)

    def policy_value_fn(self, state):
        encoded_state = np.expand_dims(encode_board(state), axis=0)
        policy, value = self.model.predict(encoded_state)
        return policy[0], value[0]

    def act(self, state):
        move = self.mcts.get_move(state)
        if move is None:
            logging.error("No valid move found by MCTS.")
            return None  # or handle the case appropriately
        return move

    def train_agent(self, num_games):
        logging.info(f"Starting training session for {num_games} games")
        for i in range(num_games):
            logging.info(f"Game {i + 1} started")
            board = chess.Board()
            states, mcts_probs, values = [], [], []

            while not board.is_game_over():
                move = self.act(board)
                if move is None:
                    break  # Handle invalid move case
                states.append(encode_board(board))
                mcts_probs.append(self.mcts.get_move_probs(board))
                board.push(move)
                values.append(evaluate_board(board))

                if board.is_game_over():
                    winner = 1 if board.result() == '1-0' else -1
                    for state, mcts_prob in zip(states, mcts_probs):
                        self.model.fit(np.expand_dims(state, axis=0), [mcts_prob, winner], epochs=1, verbose=0)

def evaluate_board(board):
    if board.is_checkmate():
        return 100 if board.turn == chess.BLACK else -100
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10000
    }
    evaluation = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            evaluation += value if piece.color == chess.WHITE else -value
    return evaluation

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    model = alphazero_model(input_shape=(8, 8, 12))
    agent = AlphaZeroAgent(model)
    agent.train_agent(EPISODES)
