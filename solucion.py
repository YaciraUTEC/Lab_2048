from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class Agent:

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.search_depth = 4 

        self.W_EMPTY     = 100.0
        self.W_SMOOTH    = 3.0
        self.W_MONO      = 10.0
        self.W_MAX       = 100.0
        self.W_POS       = 1.0

        self.TARGET_CORNER = (0, 0)
        self._pos_weights = np.array(
            [
                [65536, 32768, 16384, 8192],
                [512,   1024,  2048,  4096],
                [256,   128,   64,    32],
                [1,     2,     4,     8],
            ],
            dtype=np.float32,
        )

        self._log2 = {0: 0}
        for i in range(1, 18):
            self._log2[2 ** i] = i

        self._tt: Dict[Tuple[bytes, int, bool], float] = {}


    def act(self, board: np.ndarray, legal_actions: List[str]) -> str:
        if not legal_actions:
            return "up"

        if len(legal_actions) == 1:
            return legal_actions[0]

        self._tt.clear()

        best_action = legal_actions[0]
        best_score = -float("inf")

        for action in legal_actions:
            new_board, _, moved = self._simulate_move(board.copy(), action)
            if not moved:
                continue
            score = self._expectimax(new_board, self.search_depth - 1, is_chance=True)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _expectimax(self, board: np.ndarray, depth: int, is_chance: bool) -> float:
        key = (board.tobytes(), depth, is_chance)
        cached = self._tt.get(key)
        if cached is not None:
            return cached

        if depth == 0 or self._is_terminal(board):
            val = self._evaluate(board)
            self._tt[key] = val
            return val

        if is_chance:

            empties = list(zip(*np.where(board == 0)))
            if not empties:
                val = self._evaluate(board)
                self._tt[key] = val
                return val

            total = 0.0
            prob_per_cell = 1.0 / len(empties)

            for (r, c) in empties:
                for tile, prob in ((2, 0.9), (4, 0.1)):
                    board[r, c] = tile
                    total += prob_per_cell * prob * self._expectimax(
                        board, depth - 1, is_chance=False
                    )
                    board[r, c] = 0

            self._tt[key] = total
            return total


        best = -float("inf")
        actions = ("up", "down", "left", "right")

        for action in actions:
            new_board, _, moved = self._simulate_move(board.copy(), action)
            if not moved:
                continue
            val = self._expectimax(new_board, depth - 1, is_chance=True)
            if val > best:
                best = val

        out = best if best > -float("inf") else self._evaluate(board)
        self._tt[key] = out
        return out

    def _evaluate(self, board: np.ndarray) -> float:
        score = 0.0


        log_board = np.zeros_like(board, dtype=np.float32)
        mask = board > 0
        if np.any(mask):
            log_board[mask] = np.log2(board[mask]).astype(np.float32)


        empty_count = np.count_nonzero(board == 0)
        score += self.W_EMPTY * empty_count


        smooth = self._smoothness_fast(log_board)
        score += self.W_SMOOTH * smooth


        score += self.W_MONO * self._monotonicity(log_board)

        if log_board.shape == self._pos_weights.shape:
            positional_score = float(np.sum(board * self._pos_weights))
            score += self.W_POS * positional_score


        max_val = int(board.max())
        tr, tc = self.TARGET_CORNER
        max_log = float(np.log2(max_val)) if max_val > 0 else 0.0

        if max_val > 0:
            corner_bonus = max_val * max_val
            if int(board[tr, tc]) == max_val:
                score += self.W_MAX * corner_bonus
            else:

                score -= self.W_MAX * corner_bonus * 0.5

        return score

    def _smoothness_fast(self, log_board: np.ndarray) -> float:
        mask = log_board > 0
        mask = log_board > 0
        smooth = 0.0
        for r in range(log_board.shape[0]):
            row = log_board[r, :]
            row_mask = mask[r, :]
            for c in range(log_board.shape[1] - 1):
                if row_mask[c] and row_mask[c + 1]:
                    smooth -= abs(row[c] - row[c + 1])

        for c in range(log_board.shape[1]):
            col = log_board[:, c]
            col_mask = mask[:, c]
            for r in range(log_board.shape[0] - 1):
                if col_mask[r] and col_mask[r + 1]:
                    smooth -= abs(col[r] - col[r + 1])
        
        return smooth

    def _monotonicity(self, log_board: np.ndarray) -> float:
        score = 0.0
        for r in range(log_board.shape[0]):
            for c in range(log_board.shape[1] - 1):
                curr = log_board[r, c]
                next_val = log_board[r, c + 1]
                
                if curr == 0 or next_val == 0:
                    continue
                    
                if r % 2 == 0:
                    if curr > next_val:
                        score += (curr - next_val) * 2.0
                    else:
                        score -= (next_val - curr) * 3.0
                else:
                    if curr < next_val:
                        score += (next_val - curr) * 2.0
                    else:
                        score -= (curr - next_val) * 3.0
        for r in range(log_board.shape[0] - 1):
            for c in range(log_board.shape[1]):
                curr = log_board[r, c]
                below = log_board[r + 1, c]
                
                if curr == 0 or below == 0:
                    continue
                if c == 0:
                    if curr > below:
                        score += (curr - below) * 1.5
                elif c == log_board.shape[1] - 1:
                    if curr > below:
                        score += (curr - below) * 1.5
        
        return score


    def _simulate_move(self, board: np.ndarray, action: str):
        reward = 0
        moved = False
        if action in ("left", "right"):
            for i in range(board.shape[0]):
                row = board[i, :]
                if action == "right":
                    row = row[::-1]
                new_row, m, gained = self._merge_line(row)
                if action == "right":
                    new_row = new_row[::-1]
                if m:
                    moved = True
                reward += gained
                board[i, :] = new_row
        else:
            for j in range(board.shape[1]):
                col = board[:, j]
                if action == "down":
                    col = col[::-1]
                new_col, m, gained = self._merge_line(col)
                if action == "down":
                    new_col = new_col[::-1]
                if m:
                    moved = True
                reward += gained
                board[:, j] = new_col

        return board, reward, moved

    def _merge_line(self, line: np.ndarray):
        original = line.copy()
        nonzero = original[original != 0].tolist()
        merged = []
        reward = 0
        i = 0
        while i < len(nonzero):
            if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                v = nonzero[i] * 2
                merged.append(v)
                reward += v
                i += 2
            else:
                merged.append(nonzero[i])
                i += 1
        new_line = np.zeros_like(original)
        new_line[:len(merged)] = merged
        moved = not np.array_equal(new_line, original)
        return new_line, moved, reward

    def _is_terminal(self, board: np.ndarray) -> bool:
        if np.any(board == 0):
            return False
        for r in range(board.shape[0]):
            for c in range(board.shape[1] - 1):
                if board[r, c] == board[r, c + 1]:
                    return False
        for r in range(board.shape[0] - 1):
            for c in range(board.shape[1]):
                if board[r, c] == board[r + 1, c]:
                    return False
        return True