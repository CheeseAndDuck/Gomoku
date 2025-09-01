from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class RuleBasedPlayer: # 规则驱动的玩家

    def __init__(self, weight_my_chain: float = 10.0, weight_block_opp: float = 8.0, weight_center: float = 0.1):
        self.weight_my_chain = weight_my_chain
        self.weight_block_opp = weight_block_opp
        self.weight_center = weight_center

    def getAction(self, board, flag_is_train: bool,board_state):
        """
        - move: 选中的落子位置 (int)
        - move_probs: 大小为 W*H 的概率分布，规则引擎为确定性，选中位置为 1，其余为 0。
        """
        width, height = board.width, board.height
        n_in_row = board.n_in_row
        current_player = board.current_player
        opponent = board.players[0] if current_player == board.players[1] else board.players[1]
        availables: List[int] = list(board.availables)

        move_probs = np.zeros(width * height, dtype=np.float32)
        if not availables:
            return None, move_probs

        # 立即获胜
        for move in availables:
            if self._is_win_if_place(board.states, width, height, n_in_row, current_player, move):
                move_probs[move] = 1.0
                return move, move_probs

        # 立即防守
        threat_moves: List[int] = []
        for move in availables:
            if self._is_win_if_place(board.states, width, height, n_in_row, opponent, move):
                threat_moves.append(move)
        if threat_moves:
            # 多个威胁点时，选一个启发式最优的堵点
            best_block = max(
                threat_moves,
                key=lambda m: self._heuristic_score(board.states, width, height, n_in_row, current_player, opponent, m),
            )
            move_probs[best_block] = 1.0
            return best_block, move_probs

        # 启发式评分
        best_move = max(
            availables,
            key=lambda m: self._heuristic_score(board.states, width, height, n_in_row, current_player, opponent, m),
        )
        move_probs[best_move] = 1.0
        return best_move, move_probs

    def resetMCTS(self):
        return None

    def _heuristic_score(
        self,
        states: Dict[int, int],
        width: int,
        height: int,
        n_in_row: int,
        me: int,
        opp: int,
        move: int,
    ) -> float:
        r, c = self._pos_to_rc(move, width)
        my_len = self._max_connected_length(states, width, height, me, r, c)
        opp_len = self._max_connected_length(states, width, height, opp, r, c)

        # 中心偏好（越靠近中心越好）
        center_r = (height - 1) / 2.0
        center_c = (width - 1) / 2.0
        dist2_center = (r - center_r) ** 2 + (c - center_c) ** 2

        score = (
            self.weight_my_chain * my_len
            - self.weight_block_opp * opp_len
            - self.weight_center * dist2_center
        )
        # 若该步即可形成 >= n 连子，显著加分
        if my_len >= n_in_row:
            score += 1000.0
        return float(score)

    def _is_win_if_place(
        self,
        states: Dict[int, int],
        width: int,
        height: int,
        n_in_row: int,
        player: int,
        move: int,
    ) -> bool:
        r, c = self._pos_to_rc(move, width)
        length = self._max_connected_length(states, width, height, player, r, c)
        return length >= n_in_row

    def _max_connected_length(
        self,
        states: Dict[int, int],
        width: int,
        height: int,
        player: int,
        r: int,
        c: int,
    ) -> int:
        directions: List[Tuple[int, int]] = [(1, 0), (0, 1), (1, 1), (1, -1)]
        best = 1
        for dr, dc in directions:
            count = 1
            # 正向
            count += self._count_in_direction(states, width, height, player, r, c, dr, dc)
            # 反向
            count += self._count_in_direction(states, width, height, player, r, c, -dr, -dc)
            if count > best:
                best = count
        return best

    def _count_in_direction(
        self,
        states: Dict[int, int],
        width: int,
        height: int,
        player: int,
        r: int,
        c: int,
        dr: int,
        dc: int,
    ) -> int:
        cnt = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < height and 0 <= cc < width:
            pos = rr * width + cc
            # 将当前落子也视为己方（假设落子），仅将空位计为非己方
            if states.get(pos, 0) == player:
                cnt += 1
                rr += dr
                cc += dc
            else:
                break
        return cnt

    @staticmethod
    def _pos_to_rc(pos: int, width: int) -> Tuple[int, int]:
        return pos // width, pos % width


__all__ = ["RuleBasedPlayer"]


