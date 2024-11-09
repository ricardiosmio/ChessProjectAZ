class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self._node = {}

    def _playout(self, state):
        current_state = state.copy()
        path = []
        while not current_state.is_game_over():
            legal_moves = list(current_state.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_state.push(move)
            path.append(move)
            if current_state.fen() not in self._node:
                self._node[current_state.fen()] = 1
            else:
                self._node[current_state.fen()] += 1
        return path

    def get_move(self, state):
        for _ in range(self.n_playout):
            self._playout(state)
        legal_moves = list(state.legal_moves)
        if legal_moves:
            move_visits = [(move, self._node.get(state.fen(), 0)) for move in legal_moves]
            best_move = max(move_visits, key=lambda x: x[1])[0]
            return best_move
        return None

    def get_move_probs(self, state):
        move_probs = np.zeros(4672)
        legal_moves = list(state.legal_moves)
        
        for i, move in enumerate(legal_moves):
            state.push(move)
            move_probs[i] = self._node.get(state.fen(), 1)
            state.pop()

        # Normalize the probabilities
        move_probs /= move_probs.sum()

        return move_probs

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
        return move

    def train_agent(self, num_games):
        logging.info(f"Starting training session for {num_games} games")
        for i in range(num_games):
            logging.info(f"Game {i + 1} started")
            board = chess.Board()
            states, mcts_probs, values = [], [], []

            while not board.is_game_over():
                move = self.act(board)
                states.append(encode_board(board))
                mcts_probs.append(self.mcts.get_move_probs(board))
                board.push(move)
                values.append(evaluate_board(board))

                if board.is_game_over():
                    winner = 1 if board.result() == '1-0' else -1
                    for state, mcts_prob in zip(states, mcts_probs):
                        self.model.fit(np.expand_dims(state, axis=0), [mcts_prob, winner], epochs=1, verbose=0)
        self.save_model()

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save('models/chess_model.keras')
        logging.info("Model saved to models/chess_model.keras")

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
