import copy
import numpy as np
import torch
import torch.nn.functional as F

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode():
    def __init__(self, parent, prior_p):
        self.NUM = 1
        self.father = parent  # 父节点
        self.children = {}  # 孩子节点
        self.N_visits = 0  # 该节点的访问次数
        self.Q = 0  # 节点的总收益(V) / 总访问次数(N)
        self.U = 0  # 神经网络学习的目标：U 是正比于概率P，反比于访问次数N
        self.P = prior_p  # 走某一步棋(a)的先验概率

    def getValue(self, factor):
        self.U = (factor * self.P * np.sqrt(self.father.N_visits) / (1 + self.N_visits))
        return self.Q + self.U

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, factor):
        return max(self.children.items(), key=lambda act_node: act_node[1].getValue(factor))

    def update(self, leaf_value):
        self.N_visits += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.N_visits

    def updateRecursive(self, leaf_value):
        if self.father:
            self.NUM = 0
            for i in list(self.children.items()):
                self.NUM += i[1].NUM
            self.father.updateRecursive(-leaf_value)
        self.update(leaf_value)

    def isLeaf(self):
        return self.children == {}

    def isRoot(self):
        return self.father is None

    def __str__(self):
        return "Node(" + str(self.NUM) + ',' + str(len(self.children)) + ')'

class MCTS():
    def __init__(self, policy_NN, factor=5, simulations=100):
        self.root = TreeNode(None, 1.0)  # 初始化根节点
        self.policy_NN = policy_NN  # 神经网络
        self.fator = factor  # factor 是一个从0到正无穷的调节因子
        self.simulations = simulations  # 每次模拟推演simulation的数量

    def playout(self, state):
        node = self.root
        while True:
            if node.isLeaf():
                break
            action, node = node.select(self.fator)
            state.do_move(action)
        with torch.no_grad():
            action_probs, leaf_value = self.policy_NN(state)
            action_probs = action_probs.squeeze(0).cpu().numpy()
            leaf_value = leaf_value.squeeze(0).item()

        gameOver, winner = state.gameIsOver()

        if not gameOver:
            available_moves = state.getAvailableMoves()
            action_probs = [(move, action_probs[move]) for move in available_moves]
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.getCurrentPlayer() else -1.0

        node.updateRecursive(-leaf_value)

    def getMoveProbs(self, state, flag_is_train):
        exploration = 1.0 if flag_is_train else 1e-3

        for _ in range(self.simulations):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)

        act_visits = [(act, node.N_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / exploration * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def updateMCTS(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.father = None
        else:
            self.root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
