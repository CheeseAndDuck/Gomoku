"""
该代码定义了蒙特卡洛搜索树的决策算法，
"""
import copy
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import random
import math

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode():  # 蒙特卡洛搜索树的节点类

    def __init__(self, parent, prior_p):
        self.NUM = 1
        self.father = parent  # 父节点
        self.children = {}  # 子节点
        self.N_visits = 0  # 该节点的访问次数
        self.Q = 0  # 节点平均价值（累积更新）
        self.U = 0  # UCB探索项
        self.P = prior_p  # 先验概率（来自策略网络)
        self.immediate_reward = 0.0  # 存储当前节点的即时奖励
    

    def getValue(self, factor):  # 计算节点价值
        self.U = (factor * self.P * np.sqrt(self.father.N_visits) / (1 + self.N_visits))
        return self.Q + self.U

    def select(self, factor):  # 选择
        return max(self.children.items(), key=lambda act_node: act_node[1].getValue(factor))
    
    def expand(self, action_priors):  # 扩展
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def update(self, leaf_value):  # 更新节点访问次数和平均价值
        self.N_visits += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.N_visits
    
    def updateRecursive(self, leaf_value):  # 回溯
        if self.father:
            self.NUM = 0
            for i in list(self.children.items()):
                self.NUM += i[1].NUM
            self.father.updateRecursive(-leaf_value)  # 对手视角价值取反
        self.update(leaf_value)

    def isLeaf(self): # 判断是否为叶子节点
        return self.children == {}

    def isRoot(self):  # 判断是否为根节点
        return self.father is None

    def __str__(self):
        return f"Node(子节点数={len(self.children)}, 访问次数={self.N_visits})"


# 定义蒙特卡洛搜索树
class MCTS():  
    def __init__(self, policy_NN, value_net, factor=5, simulations=1005,max_simulation_depth=10):
        self.root = TreeNode(None, 1.0)  # 根节点初始化
        self.policy_NN = policy_NN  # 策略网络（输入状态，输出动作概率）
        self.value_net = value_net  # 价值网络（输入状态，输出状态价值）
        self.fator = factor  # UCB探索系数（平衡探索与利用）
        self.simulations = simulations  # 每次决策的模拟次数
        self.max_simulation_depth = max_simulation_depth  # 模拟的最大深度

    def playout(self, state):  # 推演过程：选择→扩展→模拟→回溯
        node = self.root
        path = []  # 记录推演路径：(节点, 动作)

        # 选择
        while not node.isLeaf():
            action, node = node.select(self.fator)
            if action >= state.width * state.height or action < 0:
                raise ValueError(f"无效动作 {action}，棋盘尺寸 {state.width}x{state.height}")
            state.do_move(action)
            state_copy = copy.deepcopy(state)
            path.append((node, action,state_copy))
         
        # 扩展
        with torch.no_grad():
            action_probs = self.policy_NN(state)  # 策略网络输出概率
            action_probs = action_probs.squeeze(0).cpu().numpy() if isinstance(action_probs, torch.Tensor) else action_probs

        # 模拟
        simulated_value = self._simulate(state,self.max_simulation_depth)

        # 终局判断
        gameOver, winner = state.gameIsOver() 
        if not gameOver:
            # 扩展
            available_moves = state.getAvailableMoves() if hasattr(state, 'getAvailableMoves') else []
            action_probs_filtered = [(move, action_probs[move]) for move in available_moves if
                                     0 <= move < len(action_probs)]
            node.expand(action_probs_filtered)
        else:
            # 终局价值修正：赢=1，输=-1，平=0（基于当前玩家视角）
            simulated_value = 1.0 if winner == state.getCurrentPlayer() else (-1.0 if winner != -1 else 0.0)

        # 回溯
        self._backpropagate(path, simulated_value)

    def _simulate(self, state,max_simulation_depth):  # 模拟：70%概率使用启发式规则，30%概率随机
        state_copy = copy.deepcopy(state)
        move_count = 0
        max_simulation_depth = max_simulation_depth
        
        while move_count < max_simulation_depth:
            game_over, winner = state_copy.gameIsOver()
            if game_over:
                # 从当前玩家视角返回结果
                if winner == -1:
                    return 0.0  # 平局
                elif winner == state.getCurrentPlayer():
                    return 1.0  # 获胜
                else:
                    return -1.0  # 失败
            available_moves = state_copy.getAvailableMoves()
            if not available_moves:
                return 0.0
            # 70%使用启发式规则，30%使用随机选择
            if random.random() < 0.7:
                move = self._heuristic_choice(state_copy)
            else:
                move = random.choice(available_moves)
            state_copy.do_move(move)
            move_count += 1
        
        with torch.no_grad():
            state_value = self.value_net.evaluate(state_copy)
            if isinstance(state_value,torch.Tensor):
                state_value = state_value.squeeze(0).item()
            return state_value


    def _backpropagate(self, path, simulated_value): # 回溯
        if not path:
            return
        cumulative_value = simulated_value
         # 先处理叶子节点（路径最后一个元素）
        leaf_node, leaf_action,leaf_satate = path[-1]
        leaf_immediate_reward = leaf_node.immediate_reward
        # 计算叶子节点的组合价值：
        leaf_combined_value = self.get_node_value(
            state=leaf_satate,
            immediate_reward=leaf_immediate_reward
        )
        leaf_node.update(leaf_combined_value)  # 更新叶子节点价值
        cumulative_value = -leaf_combined_value  # 对手视角价值取反

        for i in range(len(path)-2,-1,-1):
            node, action, state = path[i]
            # 计算当前节点的组合价值
            node_combined_value = self.get_node_value(
                state=state,  # 使用保存的状态
                immediate_reward=node.immediate_reward
            )
            node.update(node_combined_value)  # 更新当前节点价值
            cumulative_value = -node_combined_value  # 对手视角取反

        # 根节点更新（若路径仅含叶子节点）
        if len(path) == 1:
            self.root.update(leaf_combined_value)

    def _max_connected_length(self, states, width, height, player, r, c):  # 计算指定位置（r,c）对应玩家的最大连子长度
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_len = 1
        for dr, dc in directions:
            forward_len = self._count_in_direction(states, width, height, player, r, c, dr, dc)  # 正向计数（如向右、向下）
            backward_len = self._count_in_direction(states, width, height, player, r, c, -dr, -dc)  # 反向计数（如向左、向上）
            current_len = forward_len + backward_len + 1  # 总连子长度 = 正向 + 反向 + 1（当前落子）
            max_len = max(max_len, current_len)
        return min(max_len, 5) # 最多五连起子

    def _heuristic_choice(self, state):  #  启发式移动选择：优先选择能形成连子或阻断对方的棋子
        width = state.width
        height = state.height
        n_in_row = state.n_in_row
        current_player = state.getCurrentPlayer()
        opponent = 3 - current_player
        availables = state.getAvailableMoves()
        
        # 立即获胜检查
        for move in availables:
            if self._is_win_if_place(state.states, width, height, n_in_row, current_player, move):
                return move
        
        # 立即防守检查
        threat_moves = []
        for move in availables:
            if self._is_win_if_place(state.states, width, height, n_in_row, opponent, move):
                threat_moves.append(move)
        if threat_moves:
            return max(threat_moves, key=lambda m: self._heuristic_score(
                state.states, width, height, n_in_row, current_player, opponent, m))
        
        # 启发式评分选择
        best_move = max(
            availables,
            key=lambda m: self._heuristic_score(
                state.states, width, height, n_in_row, current_player, opponent, m))
        return best_move

    def _heuristic_score(self, states, width, height, n_in_row, me, opp, move):  # 计算移动的启发式分数
        r, c = move // width, move % width
        center_r, center_c = (height - 1) / 2.0, (width - 1) / 2.0
        
        # 计算四个方向的连子潜力
        my_potential = 0
        opp_potential = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            # 计算我方的
            my_len = self._count_in_direction(states, width, height, me, r, c, dr, dc)
            my_len += self._count_in_direction(states, width, height, me, r, c, -dr, -dc)
            my_potential = max(my_potential, my_len + 1)
            
            # 计算对手的
            opp_len = self._count_in_direction(states, width, height, opp, r, c, dr, dc)
            opp_len += self._count_in_direction(states, width, height, opp, r, c, -dr, -dc)
            opp_potential = max(opp_potential, opp_len + 1)
        
        # 中心距离惩罚
        dist_to_center = math.sqrt((r - center_r)**2 + (c - center_c)**2)
        center_penalty = dist_to_center * 0.1  # 距离中心越远，分数越低
        
        score = (my_potential * 5.0) + (opp_potential * 3.0) - center_penalty  # 最终得分
        
        # 形成n-1连子的额外奖励
        if my_potential >= n_in_row - 1:
            score += 100.0
        return score

    def _is_win_if_place(self, states, width, height, n_in_row, player, move):  # 检查在此位置落子是否能立即获胜
        r, c = move // width, move % width
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            count += self._count_in_direction(states, width, height, player, r, c, dr, dc)  # 正向计数
            count += self._count_in_direction(states, width, height, player, r, c, -dr, -dc)  # 反向计数
            if count >= n_in_row:
                return True
        return False

    def _count_in_direction(self, states, width, height, player, r, c, dr, dc):  # 计算指定方向上的连续棋子数
        count = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < height and 0 <= nc < width:
            pos = nr * width + nc
            if states.get(pos, -1) == player:
                count += 1
                nr += dr
                nc += dc
            else:
                break
        return count

    def get_node_value(self, state, immediate_reward):  # 结合价值网络计算节点价值
        state_value = 0.0
        if state is not None and hasattr(self.value_net, 'evaluate'):
            with torch.no_grad():
                eval_result = self.value_net.evaluate(state)
                if isinstance(eval_result, torch.Tensor):
                    state_value = eval_result.squeeze(0).item()
                else:
                    state_value = eval_result
        return state_value + immediate_reward
    

    def getMoveProbs(self, state, flag_is_train, board_state):  # 基于模拟次数统计访问频率获取落子动作与概率
        exploration = 1.0 if flag_is_train else 1e-3  # 探索系数：训练时保留探索，对战时减少探索
        for _ in range(self.simulations):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)

        valid_acts = [act for act in self.root.children.keys() if act in state.getAvailableMoves()]
        act_visits = [(act, self.root.children[act].N_visits) for act in valid_acts]
        if not act_visits:
            return [], np.array([])  # 棋盘满
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / exploration * np.log(np.array(visits) + 1e-10))  # 基于访问次数计算动作概率
        return acts, act_probs

    def updateMCTS(self, move):  # 更新MCTS树
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.father = None  # 新根节点无父节点
        else:
            self.root = TreeNode(None, 1.0)  # 动作不在子节点中，重新初始化根节点

    def __str__(self):
        return f"MCTS(模拟次数={self.simulations}, 根节点子节点数={len(self.root.children)})"

