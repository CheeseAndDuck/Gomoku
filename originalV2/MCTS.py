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


class TreeNode(): # 蒙特卡洛搜索树的节点类

    def __init__(self, parent, prior_p):
        self.NUM = 1
        self.father = parent  # 父节点
        self.children = {}  # 子节点
        self.N_visits = 0  # 该节点的访问次数
        self.Q = 0  # 节点平均价值（累积更新）
        self.U = 0  # UCB探索项
        self.P = prior_p  # 先验概率（来自策略网络）
        self.immediate_reward = 0.0  # 存储当前节点的即时奖励
    

    def getValue(self, factor):  # 计算节点价值
        self.U = (factor * self.P * np.sqrt(self.father.N_visits) / (1 + self.N_visits))
        return self.Q + self.U

    def expand(self, action_priors): # 扩展叶子节点
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, factor):  # 选择
        return max(self.children.items(), key=lambda act_node: act_node[1].getValue(factor))

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

    def isLeaf(self):
        return self.children == {}

    def isRoot(self):
        return self.father is None

    def __str__(self):
        return f"Node(子节点数={len(self.children)}, 访问次数={self.N_visits}, 即时奖励={self.immediate_reward:.3f})"


 # 蒙特卡洛搜索树
class MCTS(): 
    def __init__(self, policy_NN, value_net, factor=5, simulations=100, gamma=0.95, max_simulation_depth=10):
        self.root = TreeNode(None, 1.0)  # 根节点初始化
        self.policy_NN = policy_NN  # 策略网络（输入状态，输出动作概率）
        self.value_net = value_net  # 价值网络（输入状态，输出状态价值）
        self.fator = factor  # UCB探索系数（平衡探索与利用）
        self.simulations = simulations  # 每次决策的模拟次数
        self.gamma = gamma  # 累积折扣因子（远期奖励衰减）
        self.max_simulation_depth = max_simulation_depth  # 最大模拟深度

        # 动态α参数：控制价值网络(V)与累积奖惩(R)的权重
        self.alpha = 0.2  # 初始值（偏向累积奖惩：1-α=0.8）
        self.alpha_growth_rate = 0.005  # 每局训练后α的增长速率（逐渐偏向价值网络）
        self.max_alpha = 0.8  # α最大值（避免过度依赖价值网络）

        # 奖惩参数配置
        device = next(self.value_net.value_net.parameters()).device if hasattr(self.value_net,
                                                                               'value_net') else torch.device('cpu')
        self.reward_params = {
            # 自身连子奖励：0=无连子，1=2连，2=3连，3=4连，4=5连
            'my_chain': torch.tensor([0.0, 0.0, 0.3, 0.8, 1.5], device=device),
            # 阻止对手连子奖励：0=无阻止，1=阻2连，2=阻3连，3=阻4连，4=阻5连
            'opp_chain': torch.tensor([0.0, 0.0, 0.2, 0.5, 1.0], device=device),
            # 自身被阻止连子惩罚：0=无惩罚，1=2连被阻，2=3连被阻，3=4连被阻，4=5连被阻
            'blocked_penalty': torch.tensor([0.0, 0.0, -0.2, -0.5, -1.0], device=device),
            # 落子数惩罚参数
            'move_count_penalty': {
                'base_factor': 0.01,  # 基础惩罚系数
                'max_moves': 225,     # 最大落子数（15x15棋盘）
                'alpha_dependence': 0.7  # α值对惩罚的影响系数（0-1）
            }
        }
        
    def playout(self, state):  # 推演过程：选择→扩展→模拟→回溯
        node = self.root
        path = [] 

        # 选择
        while not node.isLeaf():
            action, node = node.select(self.fator)
            if action >= state.width * state.height or action < 0:
                raise ValueError(f"无效动作 {action}，棋盘尺寸 {state.width}x{state.height}")
            state.do_move(action)  # 执行动作更新状态
            state_copy = copy.deepcopy(state)
            path.append((node, action,state_copy))
            
            # 计算当前动作的即时奖励并存储到节点
            immediate_reward = self._calculate_immediate_reward(state, action)
            node.immediate_reward = immediate_reward

        # 扩展
        with torch.no_grad():
            action_probs = self.policy_NN(state)  # 策略网络输出动作概率
            action_probs = action_probs.squeeze(0).cpu().numpy() if isinstance(action_probs, torch.Tensor) else action_probs

        # 模拟
        simulated_value = self._simulate(state)
        
        # 终局判断
        gameOver, winner = state.gameIsOver()
        if not gameOver:
            available_moves = state.getAvailableMoves() if hasattr(state, 'getAvailableMoves') else []
            action_probs_filtered = [(move, action_probs[move]) for move in available_moves if
                                     0 <= move < len(action_probs)]
            node.expand(action_probs_filtered)
        else:
            # 终局价值修正：赢=1，输=-1，平=0（基于当前玩家视角）
            simulated_value = 1.0 if winner == state.getCurrentPlayer() else (-1.0 if winner != -1 else 0.0)

        # 回溯
        self._backpropagate(path, simulated_value)

    def _simulate(self, state):  # 模拟：70%概率使用启发式规则，30%概率随机
        state_copy = copy.deepcopy(state)
        move_count = 0
        max_simulation_depth = 10  # 最大模拟深度
        
        while move_count < max_simulation_depth:
            game_over, winner = state_copy.gameIsOver()
            if game_over:
                # 从当前玩家视角返回结果
                if winner == -1:  # 平局
                    return 0.0
                elif winner == state.getCurrentPlayer():  # 获胜
                    return 1.0
                else:  # 失败
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
        
        # 达到最大模拟深度后，使用价值网络评估当前状态
        with torch.no_grad():
            state_value = self.value_net.evaluate(state_copy)
            if isinstance(state_value, torch.Tensor):
                state_value = state_value.squeeze(0).item()
            return state_value


    def _backpropagate(self, path, simulated_value):  # 回溯：按折扣因子累积奖惩，使用模拟结果更新节点价值
        if not path:
            return

        # 从叶子节点向根节点反向遍历
        cumulative_value = simulated_value
        
        # 先处理叶子节点（路径最后一个元素）
        leaf_node, leaf_action,leaf_satate = path[-1]
        leaf_immediate_reward = leaf_node.immediate_reward

        # 计算叶子节点的组合价值：α*V + (1-α)*(R + γ*V未来)
        leaf_combined_value = self.get_node_value(
            state=leaf_satate,
            immediate_reward=leaf_immediate_reward,
            future_value=cumulative_value
        )
        leaf_node.update(leaf_combined_value)  # 更新叶子节点价值
        cumulative_value = -leaf_combined_value  # 对手视角价值取反

        for i in range(len(path)-2,-1,-1):
            node, action, state = path[i]
            # 计算当前节点的组合价值
            node_combined_value = self.get_node_value(
                state=state,  # 使用保存的状态
                immediate_reward=node.immediate_reward,
                future_value=cumulative_value
            )
            node.update(node_combined_value)  # 更新当前节点价值
            cumulative_value = -node_combined_value  # 对手视角取反

        # 根节点更新（若路径仅含叶子节点）
        if len(path) == 1:
            self.root.update(leaf_combined_value)

    def _calculate_immediate_reward(self, state, action):
        """计算即时奖励：自身连子 + 阻敌 + 被阻惩罚 + 落子数惩罚"""
        if action is None or not hasattr(state, 'states'):
            return 0.0  # 无动作或无效状态，奖励为0

        # 基础参数获取
        width = state.width if hasattr(state, 'width') else 15
        height = state.height if hasattr(state, 'height') else 15
        current_player = state.getCurrentPlayer() if hasattr(state, 'getCurrentPlayer') else 1
        opponent = 2 if current_player == 1 else 1
        r, c = action // width, action % width  # 落子坐标
        states = state.states  # 棋盘落子状态

        # 自身连子奖励
        my_chain_len = self._max_connected_length(states, width, height, current_player, r, c)
        my_chain_level = min(my_chain_len - 1, 4)  # 2连→等级1，5连→等级4
        my_chain_reward = self.reward_params['my_chain'][my_chain_level].item()

        # 阻止对手连子奖励
        opp_chain_len = self._max_connected_length(states, width, height, opponent, r, c)
        opp_chain_level = min(opp_chain_len - 1, 4)
        opp_chain_reward = self.reward_params['opp_chain'][opp_chain_level].item()

        # 自身被阻止连子惩罚
        blocked_len = self._check_blocked_chain(states, width, height, current_player, r, c)
        blocked_level = min(blocked_len - 1, 4)
        blocked_penalty = self.reward_params['blocked_penalty'][blocked_level].item()

        # 落子数惩罚（步数越多惩罚越大，随α增大而减弱
        move_count_penalty = 0.0
        if hasattr(state, 'states'):
            total_moves = len(state.states)  # 当前已落子数
            max_moves = self.reward_params['move_count_penalty']['max_moves']
            base_factor = self.reward_params['move_count_penalty']['base_factor']
            alpha_dependence = self.reward_params['move_count_penalty']['alpha_dependence']
            
            # 计算落子数惩罚：基础惩罚 × (落子数/最大落子数) × (1-α)^影响系数
            move_count_ratio = total_moves / max_moves
            alpha_effect = alpha_dependence
            move_count_penalty = -base_factor * move_count_ratio * alpha_effect

        #  总即时奖励
        total_immediate_reward = (my_chain_reward + opp_chain_reward + blocked_penalty + move_count_penalty)
        return total_immediate_reward
    
    def _max_connected_length(self, states, width, height, player, r, c):  # 计算指定位置（r,c）对应玩家的最大连子长度
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_len = 1

        for dr, dc in directions:
            forward_len = self._count_in_direction(states, width, height, player, r, c, dr, dc)
            backward_len = self._count_in_direction(states, width, height, player, r, c, -dr, -dc)
            current_len = forward_len + backward_len + 1
            max_len = max(max_len, current_len)

        # 连子长度最大为5
        return min(max_len, 5)

    def _check_blocked_chain(self, states, width, height, player, r, c):  # 检查指定位置（r,c）对应玩家的连子是否被对手阻止，返回被阻止的连子长度
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_blocked_len = 1  # 初始为当前落子（无连子则无惩罚）
        opponent = 2 if player == 1 else 1

        for dr, dc in directions:
            # 正向检查：当前位置→方向延伸，是否被对手棋子阻断
            forward_blocked = self._count_blocked_in_direction(states, width, height, player, opponent, r, c, dr, dc)
            # 反向检查：当前位置→反方向延伸，是否被对手棋子阻断
            backward_blocked = self._count_blocked_in_direction(states, width, height, player, opponent, r, c, -dr, -dc)
            # 被阻止的连子长度 = 正向被阻 + 反向被阻 + 1（当前落子）
            current_blocked_len = forward_blocked + backward_blocked + 1
            max_blocked_len = max(max_blocked_len, current_blocked_len)
        return min(max_blocked_len, 5)

    def _heuristic_choice(self, state):  # 启发式移动选择：优先选择能形成连子或阻断对方的棋子
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

    def _is_win_if_place(self, states, width, height, n_in_row, player, move):  # 检查在此位置落子是否能立即获胜
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
    

    def _count_blocked_in_direction(self, states, width, height, player, opponent, r, c, dr, dc):  # 计算指定方向上，玩家连子被对手阻断的长度
        count = 0
        nr, nc = r + dr, c + dc  # 下一个位置

        # 第一步：统计当前方向上的玩家连子
        while 0 <= nr < height and 0 <= nc < width:
            pos = nr * width + nc
            if states.get(pos, -1) == player:
                count += 1
                nr += dr
                nc += dc
            else:
                break

        # 第二步：检查连子末端是否被对手棋子阻断（若连子存在且末端非空）
        if count > 0 and 0 <= nr < height and 0 <= nc < width:
            if states.get(nr * width + nc, -1) == opponent:
                return count  # 连子被对手阻断，返回连子长度
            else:
                return 0  # 连子末端是空位置，未被阻断
        else:
            return 0  # 无连子或超出棋盘，无阻断

    def get_node_value(self, state, immediate_reward, future_value):  # 计算节点组合价值：α*V(价值网络) + (1-α)*(R + γ*V未来)
        # 价值网络输出的状态价值
        state_value = 0.0
        if state is not None and hasattr(self.value_net, 'evaluate'):
            with torch.no_grad():
                eval_result = self.value_net.evaluate(state)
                if isinstance(eval_result, torch.Tensor):
                    state_value = eval_result.squeeze(0).item()
                else:
                    state_value = eval_result

        # 累积折扣奖惩：即时奖励 + 折扣因子 * 未来价值
        cumulative_reward = immediate_reward + self.gamma * future_value

        # 组合价值：平衡价值网络与累积奖惩
        combined_value = self.alpha * state_value + (1 - self.alpha) * cumulative_reward
        return combined_value

    def update_alpha(self, performance_metric=None):  # 动态更新α值
        growth_factor = 1.0
        if performance_metric is not None:
            if isinstance(performance_metric, float):
                # 若指标是价值损失：损失越小，α增长越快
                if performance_metric < 0.1:
                    growth_factor = 1.5
                elif performance_metric < 0.2:
                    growth_factor = 1.2
                else:
                    growth_factor = 0.8
            elif isinstance(performance_metric, tuple):
                # 胜率越高，α增长越快
                win_count, total_count = performance_metric
                win_rate = win_count / total_count if total_count > 0 else 0.5
                growth_factor = 1.5 if win_rate > 0.7 else (1.0 if win_rate > 0.5 else 0.8)
        self.alpha = min(self.max_alpha, self.alpha + self.alpha_growth_rate * growth_factor) # 更新α
        return self.alpha

    def getMoveProbs(self, state, flag_is_train, board_state):  # 基于模拟次数统计访问频率获取落子动作与概率
        # 探索系数：训练时保留探索，对战时减少探索
        exploration = 1.0 if flag_is_train else 1e-3

        # 执行所有模拟推演
        for _ in range(self.simulations):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)

        # 统计根节点子节点的访问次数，但只考虑有效的动作
        valid_acts = [act for act in self.root.children.keys() if act in state.getAvailableMoves()]
        act_visits = [(act, self.root.children[act].N_visits) for act in valid_acts]
        
        if not act_visits:
            return [], np.array([])

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / exploration * np.log(np.array(visits) + 1e-10))    # 基于访问次数计算动作概率

        return acts, act_probs

    def updateMCTS(self, move):  # 更新MCTS树
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.father = None  # 新根节点无父节点
        else: 
            self.root = TreeNode(None, 1.0)  # 动作不在子节点中，重新初始化根节点

    def __str__(self):
        return f"MCTS(模拟次数={self.simulations}, α={self.alpha:.3f}, 根节点子节点数={len(self.root.children)})"

