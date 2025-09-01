import copy
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import random
import math

def softmax(x):
    probs = np.exp(x - np.max(x))  # 数值稳定版softmax
    probs /= np.sum(probs)
    return probs


class TreeNode():
    """蒙特卡洛搜索树的节点类（保持结构不变，补充即时奖励存储）"""

    def __init__(self, parent, prior_p):
        self.NUM = 1
        self.father = parent  # 父节点
        self.children = {}  # 子节点：{action: TreeNode}
        self.N_visits = 0  # 该节点的访问次数
        self.Q = 0  # 节点平均价值（累积更新）
        self.U = 0  # UCB探索项
        self.P = prior_p  # 先验概率（来自策略网络)
        self.immediate_reward = 0.0  # 存储当前节点的即时奖励（新增）
    

    def getValue(self, factor):
        """计算节点价值（UCB公式不变）"""
        self.U = (factor * self.P * np.sqrt(self.father.N_visits) / (1 + self.N_visits))
        return self.Q + self.U

    def expand(self, action_priors):
        """扩展叶子节点（新增动作-概率对）"""
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, factor):
        """选择子节点（按UCB值最大原则）"""
        return max(self.children.items(), key=lambda act_node: act_node[1].getValue(factor))

    def update(self, leaf_value):
        """更新节点访问次数和平均价值（增量更新公式）"""
        self.N_visits += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.N_visits  # 避免重新计算总和

    def updateRecursive(self, leaf_value):
        """递归回溯更新（原始逻辑保留，适配新的价值计算）"""
        if self.father:
            self.NUM = 0
            for i in list(self.children.items()):
                self.NUM += i[1].NUM
            self.father.updateRecursive(-leaf_value)  # 对手视角价值取反
        self.update(leaf_value)

    def isLeaf(self):
        """判断是否为叶子节点（无子女）"""
        return self.children == {}

    def isRoot(self):
        """判断是否为根节点（无父节点）"""
        return self.father is None

    def __str__(self):
        return f"Node(子节点数={len(self.children)}, 访问次数={self.N_visits})"



class MCTS():
    """蒙特卡洛搜索树（改进版：增强模拟阶段策略）"""

    def __init__(self, policy_NN, value_net, factor=5, simulations=1005):
        self.root = TreeNode(None, 1.0)  # 根节点初始化
        self.policy_NN = policy_NN  # 策略网络（输入状态，输出动作概率）
        self.value_net = value_net  # 价值网络（输入状态，输出状态价值）
        self.fator = factor  # UCB探索系数（平衡探索与利用）
        self.simulations = simulations  # 每次决策的模拟次数

        
    def playout(self, state):
        """推演过程：从根到叶选择→扩展→模拟→反向传播"""
        node = self.root
        path = []  # 记录推演路径：[(节点, 动作), ...]

        # 1. 选择阶段：从根节点遍历到叶子节点
        while not node.isLeaf():
            action, node = node.select(self.fator)
            
            # 确保动作有效
            if action >= state.width * state.height or action < 0:
                raise ValueError(f"无效动作 {action}，棋盘尺寸 {state.width}x{state.height}")
            
            state.do_move(action)  # 执行动作更新状态
            path.append((node, action))
            

        # 2. 扩展阶段：扩展叶子节点
        with torch.no_grad():
            # 策略网络输出动作概率
            action_probs = self.policy_NN(state)
            action_probs = action_probs.squeeze(0).cpu().numpy() if isinstance(action_probs, torch.Tensor) else action_probs

        # 3. 模拟阶段：使用随机策略模拟直到游戏结束
        simulated_value = self._simulate(state)
        
        # 4. 终局判断：若未结束则扩展叶子节点
        gameOver, winner = state.gameIsOver()
        if not gameOver:
            # 扩展叶子节点（仅可用动作）
            available_moves = state.getAvailableMoves() if hasattr(state, 'getAvailableMoves') else []
            action_probs_filtered = [(move, action_probs[move]) for move in available_moves if
                                     0 <= move < len(action_probs)]
            node.expand(action_probs_filtered)
        else:
            # 终局价值修正：赢=1，输=-1，平=0（基于当前玩家视角）
            simulated_value = 1.0 if winner == state.getCurrentPlayer() else (-1.0 if winner != -1 else 0.0)

        # 5. 反向传播阶段：更新路径上所有节点的价值
        self._backpropagate(path, simulated_value)


    def _simulate(self, state):
        """增强版模拟阶段：70%概率使用启发式规则，30%概率随机"""
        state_copy = copy.deepcopy(state)
        move_count = 0
        
        while True:
            game_over, winner = state_copy.gameIsOver()
            if game_over:
                # 从当前玩家视角返回结果
                if winner == -1:  # 平局
                    return 0.0
                elif winner == state.getCurrentPlayer():  # 当前玩家获胜
                    return 1.0
                else:  # 对手获胜
                    return -1.0
                    
            available_moves = state_copy.getAvailableMoves()
            if not available_moves:
                return 0.0
                
            # 70%概率使用启发式规则，30%概率随机选择
            if random.random() < 0.7:
                move = self._heuristic_choice(state_copy)
            else:
                move = random.choice(available_moves)
                
            state_copy.do_move(move)
            move_count += 1


    def _backpropagate(self, path, simulated_value):
        """反向传播：仅用即时奖励+模拟价值（删除累积折扣）"""
        if not path:
            return  # 空路径无需回溯
        
        # 1. 处理叶子节点：用模拟价值（终局结果）替代价值网络评估
        leaf_node, _ = path[-1]
        leaf_immediate_reward = leaf_node.immediate_reward
        # 叶子节点为终局，模拟价值即真实状态价值，无需额外评估
        leaf_combined_value = self.get_node_value(
            state=None,  # 终局状态无需传参，用模拟价值替代
            immediate_reward=leaf_immediate_reward
        )
        # 修正：叶子节点需结合模拟价值（因无state时value_net无法评估）jaing
        leaf_combined_value = simulated_value
        leaf_node.update(leaf_combined_value)
        
        # 2. 处理非叶子节点：仅用即时奖励（无单独状态，价值网络无法评估）
        for node, _ in reversed(path[:-1]):
            node_immediate_reward = node.immediate_reward
            # 非叶子节点无独立状态，价值网络评估为0，仅用即时奖励
            node_combined_value = self.get_node_value(
                state=None,
                immediate_reward=node_immediate_reward
            )
            node.update(node_combined_value)
        
        # 3. 确保根节点更新（若路径仅含叶子节点）
        if len(path) == 1:
            self.root.update(leaf_combined_value)


    def _max_connected_length(self, states, width, height, player, r, c):
        """计算指定位置（r,c）对应玩家的最大连子长度（四个方向：水平/垂直/斜向）"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 右、下、右下、右上
        max_len = 1  # 至少包含当前落子

        for dr, dc in directions:
            # 正向计数（如向右、向下）
            forward_len = self._count_in_direction(states, width, height, player, r, c, dr, dc)
            # 反向计数（如向左、向上）
            backward_len = self._count_in_direction(states, width, height, player, r, c, -dr, -dc)
            # 总连子长度 = 正向 + 反向 + 1（当前落子）
            current_len = forward_len + backward_len + 1
            max_len = max(max_len, current_len)

        # 限制连子长度最大为5（五子棋规则）
        return min(max_len, 5)


    def _heuristic_choice(self, state):
        """启发式移动选择：优先选择能形成连子或阻断对方的棋子"""
        width = state.width
        height = state.height
        n_in_row = state.n_in_row
        current_player = state.getCurrentPlayer()
        opponent = 3 - current_player
        availables = state.getAvailableMoves()
        
        # 1. 立即获胜检查
        for move in availables:
            if self._is_win_if_place(state.states, width, height, n_in_row, current_player, move):
                return move
        
        # 2. 立即防守检查
        threat_moves = []
        for move in availables:
            if self._is_win_if_place(state.states, width, height, n_in_row, opponent, move):
                threat_moves.append(move)
        if threat_moves:
            return max(threat_moves, key=lambda m: self._heuristic_score(
                state.states, width, height, n_in_row, current_player, opponent, m))
        
        # 3. 启发式评分选择
        best_move = max(
            availables,
            key=lambda m: self._heuristic_score(
                state.states, width, height, n_in_row, current_player, opponent, m))
        return best_move

    def _heuristic_score(self, states, width, height, n_in_row, me, opp, move):
        """计算移动的启发式分数"""
        r, c = move // width, move % width
        center_r, center_c = (height - 1) / 2.0, (width - 1) / 2.0
        
        # 计算四个方向的连子潜力
        my_potential = 0
        opp_potential = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            # 计算我方潜力
            my_len = self._count_in_direction(states, width, height, me, r, c, dr, dc)
            my_len += self._count_in_direction(states, width, height, me, r, c, -dr, -dc)
            my_potential = max(my_potential, my_len + 1)
            
            # 计算对手潜力
            opp_len = self._count_in_direction(states, width, height, opp, r, c, dr, dc)
            opp_len += self._count_in_direction(states, width, height, opp, r, c, -dr, -dc)
            opp_potential = max(opp_potential, opp_len + 1)
        
        # 中心距离惩罚
        dist_to_center = math.sqrt((r - center_r)**2 + (c - center_c)**2)
        center_penalty = dist_to_center * 0.1  # 距离中心越远，分数越低
        
        # 组成最终分数
        score = (my_potential * 5.0) + (opp_potential * 3.0) - center_penalty
        
        # 形成n-1连子的额外奖励
        if my_potential >= n_in_row - 1:
            score += 100.0
        return score

    def _is_win_if_place(self, states, width, height, n_in_row, player, move):
        """检查在此位置落子是否能立即获胜"""
        r, c = move // width, move % width
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # 当前位置
            
            # 正向计数
            count += self._count_in_direction(states, width, height, player, r, c, dr, dc)
            # 反向计数
            count += self._count_in_direction(states, width, height, player, r, c, -dr, -dc)
            
            if count >= n_in_row:
                return True
        return False

    def _count_in_direction(self, states, width, height, player, r, c, dr, dc):
        """计算指定方向上的连续棋子数"""
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
    


    def get_node_value(self, state, immediate_reward):
        """计算节点价值：仅结合价值网络"""
        state_value = 0.0
        if state is not None and hasattr(self.value_net, 'evaluate'):
            with torch.no_grad():
                eval_result = self.value_net.evaluate(state)
                state_value = eval_result.squeeze(0).item() if isinstance(eval_result, torch.Tensor) else 0.0
        return state_value
    

    def getMoveProbs(self, state, flag_is_train, board_state):
        """获取落子动作与概率：基于模拟次数统计访问频率"""
        # 探索系数：训练时保留探索，对战时减少探索
        exploration = 1.0 if flag_is_train else 1e-3

        # 执行所有模拟推演
        for _ in range(self.simulations):
            state_copy = copy.deepcopy(state)  # 深拷贝状态，避免污染原状态
            self.playout(state_copy)

        # 统计根节点子节点的访问次数，但只考虑有效的动作
        valid_acts = [act for act in self.root.children.keys() if act in state.getAvailableMoves()]
        act_visits = [(act, self.root.children[act].N_visits) for act in valid_acts]
        
        if not act_visits:
            return [], np.array([])  # 无可用动作（棋盘满）

        acts, visits = zip(*act_visits)
        # 基于访问次数计算动作概率（softmax平滑）
        act_probs = softmax(1.0 / exploration * np.log(np.array(visits) + 1e-10))  # 加1e-10避免log(0)

        return acts, act_probs

    def updateMCTS(self, move):
        """更新MCTS树：将指定动作作为新根节点（保留历史探索信息）"""
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.father = None  # 新根节点无父节点
        else:
            # 动作不在子节点中，重新初始化根节点
            self.root = TreeNode(None, 1.0)

    def __str__(self):
        return f"MCTS(模拟次数={self.simulations}, 根节点子节点数={len(self.root.children)})"

