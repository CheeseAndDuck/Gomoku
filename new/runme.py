from __future__ import print_function
import os
import threading
import random
import numpy as np
import pickle
from tqdm import tqdm
import math
import csv
import json
import torch
from collections import deque 

from MCTS import *
from tactics import PolicyValueNet
from evaluateValue import ValueTrainer
from ruleAI import RuleBasedPlayer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Board(object):
    def __init__(self, width=15, height=15, n_in_row=5):
        self.width = width
        self.height = height
        self.states = {}  # {落子位置: 玩家}
        self.n_in_row = n_in_row  # 连子获胜数
        self.players = [1, 2]  # 玩家1和玩家2

    def initBoard(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception(f'板宽和板高不得小于{self.n_in_row}')
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))  # 可用落子位置
        self.states = {}
        self.last_move = -1  # 最后落子位置

    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]  # 当前玩家落子
            move_oppo = moves[players != self.current_player]  # 对手落子

            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0  # 最后落子标记

        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # 回合标记
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
      
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]  # 切换玩家
        self.last_move = move

    def has_a_winner(self):
        width, height = self.width, self.height
        states, n = self.states, self.n_in_row
        moved = list(set(range(width * height)) - set(self.availables))

        if len(moved) < n * 2 - 1:
            return False, -1

        for m in moved:
            h, w = m // width, m % width
            player = states[m]
            # 水平方向
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player
            # 垂直方向
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player
            # 右下对角线
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player
            # 左下对角线
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def gameIsOver(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):  # 棋盘下满
            return True, -1
        return False, -1

    def getCurrentPlayer(self):
        return self.current_player

    def getState(self):
        return self.current_state()

    def getAvailableMoves(self):
        return list(self.availables)


class Game():
    def __init__(self, width=15, height=15, n_in_row=5, flag_is_train=True):
        self.boardWidth = width
        self.boardHeight = height
        self.n_in_row = n_in_row
        self.flag_is_train = flag_is_train
        self.board = Board(width=width, height=height, n_in_row=n_in_row)

    def selfPlay(self, player, Index=0):
        self.board.initBoard()
        boards, probs, currentPlayer = [], [], []

        while True:
            move, move_probs = player.getAction(self.board, self.flag_is_train)
            boards.append(self.board.current_state())
            probs.append(move_probs)
            currentPlayer.append(self.board.current_player)
            self.board.do_move(move)
            gameOver, winner = self.board.gameIsOver()
            if gameOver:
                winners_z = np.zeros(len(currentPlayer))
                if winner != -1:
                    winners_z[np.array(currentPlayer) == winner] = 1.0
                    winners_z[np.array(currentPlayer) != winner] = -1.0
                player.resetMCTS()
                return winner, zip(boards, probs, winners_z)

    def playAIVsAI(self, player_black, player_white, max_moves: int = 225):
        self.board.initBoard(0)
        boards, probs, currentPlayer = [], [], []

        for _ in range(max_moves):
            current_player_obj = player_black if self.board.current_player == 1 else player_white
            move, move_probs = current_player_obj.getAction(self.board, flag_is_train=False)
            boards.append(self.board.current_state())
            probs.append(move_probs)
            currentPlayer.append(self.board.current_player)
            self.board.do_move(move)
            gameOver, winner = self.board.gameIsOver()
            if gameOver:
                winners_z = np.zeros(len(currentPlayer))
                if winner != -1:
                    winners_z[np.array(currentPlayer) == winner] = 1.0
                    winners_z[np.array(currentPlayer) != winner] = -1.0
                if hasattr(player_black, 'resetMCTS'):
                    player_black.resetMCTS()
                if hasattr(player_white, 'resetMCTS'):
                    player_white.resetMCTS()
                return winner, zip(boards, probs, winners_z)
        if hasattr(player_black, 'resetMCTS'):
            player_black.resetMCTS()
        if hasattr(player_white, 'resetMCTS'):
            player_white.resetMCTS()
        winners_z = np.zeros(len(currentPlayer))
        return -1, zip(boards, probs, winners_z)


class MCTSPlayer():
    def __init__(self, policy_NN):
        self.simulations = 800  # 每次行动的模拟次数
        self.factor = 2  # 探索因子（平衡先验概率和UCT值）
        self.MCTS = MCTS(policy_NN, self.factor, self.simulations)  # 初始化MCTS

    def resetMCTS(self):
        self.MCTS.updateMCTS(-1)

    def getAction(self, board, flag_is_train):
        emptySpacesBoard = board.availables
        move_probs = np.zeros(board.width * board.height)  # 初始化落子概率

        if len(emptySpacesBoard) > 0:
            acts, probs = self.MCTS.getMoveProbs(board, flag_is_train)
            move_probs[list(acts)] = probs

            if flag_is_train:
                # 训练模式：添加Dirichlet噪声增强探索
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                self.MCTS.updateMCTS(move)
            else:
                # 非训练模式：选择概率最高的落子
                move = np.random.choice(acts, p=probs)
                self.MCTS.updateMCTS(-1)  # 重置MCTS树

            return move, move_probs
        else:
            print("WARNING: 棋盘已满")


class MetaZeta(threading.Thread):
    save_ParaFreq = 10 
    MAX_Games = 30  # 总训练局数

    def __init__(self, flag_is_train=True,load_from_checkpoint=None):
        super().__init__()
        self.flag_is_train = flag_is_train
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.game = Game(width=15, height=15, n_in_row=5, flag_is_train=flag_is_train)
        self.policy_net = PolicyValueNet((4, 15, 15))  # 策略网络
        self.value_net = ValueTrainer((4, 15, 15))  # 价值网络
        self.policy_net.to(self.device)
        self.value_net.value_net.to(self.device) 
        if load_from_checkpoint:
            self.load_checkpoint(load_from_checkpoint)
        self.MCTSPlayer = MCTSPlayer(
            policy_NN=lambda s: (self.policy_net.policy_NN(s), self.value_net.evaluate(s))
        )
        self.rule_player = RuleBasedPlayer()
        self.start()


    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_net.optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_net.optimizer.state_dict(),
            'policy_trainDataPool': list(self.policy_net.trainDataPool),
            'value_trainDataPool': list(self.value_net.trainDataPool),
            'policy_LRfctor': self.policy_net.LRfctor,
            'policy_learningRate': self.policy_net.learningRate,
            'value_lr': self.value_net.lr,
            'current_game': getattr(self, 'current_game', 0)
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存到 {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"检查点文件 {checkpoint_path} 不存在，从头开始训练")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载策略网络
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.policy_net.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.policy_net.trainDataPool = deque(checkpoint.get('policy_trainDataPool', []), 
                                            maxlen=self.policy_net.trainDataPoolSize)
        self.policy_net.LRfctor = checkpoint.get('policy_LRfctor', 1.0)
        self.policy_net.learningRate = checkpoint.get('policy_learningRate', 2e-3)
        
        # 加载价值网络
        self.value_net.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.value_net.optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.value_net.trainDataPool = deque(checkpoint.get('value_trainDataPool', []), 
                                           maxlen=self.value_net.trainDataPool.maxlen)
        self.value_net.lr = checkpoint.get('value_lr', 1e-3)
        
        # 加载当前游戏进度
        self.current_game = checkpoint.get('current_game', 0)
        
        print(f"从检查点 {checkpoint_path} 加载成功")


    def run(self):
        if self.flag_is_train:
            self.train()  # 训练模式
        else:
            self.battle_mode()  # 对战模式

    def battle_mode(self):
        try:
            self.policy_net.load_model("new/models2/Version1.model")
            print("已加载训练好的模型")
        except Exception as e:
            print(f"加载模型失败: {str(e)}，使用随机初始化模型")

        num_battles = 10
        print(f"开始与规则AI对战，共{num_battles}局...")

        win_rate = self.evaluate(num_games=num_battles)
        print(f"对战完成！胜率: {win_rate:.2%}")
        try:
            with open('battle_results.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not os.path.exists('battle_results.csv'):
                    writer.writerow(['Date', 'Battles', 'Wins', 'Draws', 'Win_Rate'])
                from datetime import datetime
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    num_battles,
                    int(win_rate * num_battles),
                    int((1 - win_rate) * num_battles),
                    win_rate
                ])
        except Exception as e:
            print(f"保存对战结果失败: {str(e)}")

    def save_to_csv(self, play_data, game_index, winner, filepath='play_chess.csv'):
        seq = list(play_data)

        def serialize(arr):
            return json.dumps(np.array(arr).tolist(), ensure_ascii=False)

        rows = []
        final_state = self.game.board.current_state()  # 终局状态

        if winner == -1:
            # 平局：记录双方最后一次落子
            if len(seq) >= 1:
                idx_first = max([i for i in range(len(seq)) if i % 2 == 0])
                _, probs_first, _ = seq[idx_first]
                rows.append({
                    'game': game_index,
                    'result': 0,
                    'state': serialize(final_state),
                    'move_probs': serialize(probs_first)
                })
            if len(seq) >= 2:
                idx_second = max([i for i in range(len(seq)) if i % 2 == 1])
                _, probs_second, _ = seq[idx_second]
                rows.append({
                    'game': game_index,
                    'result': 0,
                    'state': serialize(final_state),
                    'move_probs': serialize(probs_second)
                })
        else:
            # 胜负局：记录胜负双方最后落子
            idx_win, idx_lose = None, None
            for i, (_, _, wz) in enumerate(seq):
                if wz == 1:
                    idx_win = i
                elif wz == -1:
                    idx_lose = i
            if idx_win is not None:
                _, probs_w, _ = seq[idx_win]
                rows.append({
                    'game': game_index,
                    'result': 1,
                    'state': serialize(final_state),
                    'move_probs': serialize(probs_w)
                })
            if idx_lose is not None:
                _, probs_l, _ = seq[idx_lose]
                rows.append({
                    'game': game_index,
                    'result': -1,
                    'state': serialize(final_state),
                    'move_probs': serialize(probs_l)
                })

        file_exists = os.path.exists(filepath)
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['game', 'result', 'state', 'move_probs'])
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def train(self):
        start_game = getattr(self, 'current_game', 0)
        
        for oneGame in range(start_game, self.MAX_Games):
            print(f'正在进行第{oneGame + 1}局自我对弈···')
            # 自我对弈生成数据
            winner, play_data = self.game.selfPlay(self.MCTSPlayer, Index=oneGame + 1)
            play_data = list(play_data)  # 转换为列表

            # 存储训练数据
            self.policy_net.memory(play_data)
            self.value_net.memory(play_data)

            # 保存对弈数据
            try:
                self.save_to_csv(play_data, oneGame + 1, winner)
                print('对弈数据已保存到 play_chess.csv')
            except Exception as e:
                print("对弈数据保存失败: "+str(e))

            # 训练网络
            tactic_loss = self.policy_net.update()
            value_loss = self.value_net.update()
            if tactic_loss:
                print(f"策略网络损失:{tactic_loss}")
            if value_loss:
                print(f"价值网络损失: {value_loss}")

            if (oneGame + 1) % self.save_ParaFreq == 0:
                self.policy_net.save_model(f'new/models2/tactic_{oneGame + 1}.model')
                self.value_net.save_model(f'new/models2/value_{oneGame + 1}.model')
                # 保存检查点
                self.save_checkpoint(f'new/models2/checkpoint_{oneGame + 1}.pth')
                print(f'已保存第{oneGame + 1}局模型和检查点')

        # 训练结束后最终评估
        print("训练完成，开始最终评估...")
        final_win_rate = self.evaluate(num_games=20)
        print(f"最终评估胜率{final_win_rate}")

        # 保存最终模型和检查点
        self.policy_net.save_model('new/models2/final_tactic.model')
        self.value_net.save_model('new/models2/final_value.model')
        self.save_checkpoint('new/models2/final_checkpoint.pth')
        
        # 保存最终评估结果
        try:
            with open('evaluation_results.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not os.path.exists('evaluation_results.csv'):
                    writer.writerow(['Type', 'Win_Rate'])
                writer.writerow(['Final', final_win_rate])
        except Exception as e:
            print("保存评估结果失败:")


    def evaluate(self, num_games=10):
        wins = 0
        draws = 0

        for i in range(0, num_games, 2):
            # 轮流先手
            print("评估对局" +str(((i+1))/num_games) + "（先手）")
            winner, _ = self.game.playAIVsAI(self.MCTSPlayer, self.rule_player)
            if winner == 1:
                wins += 1
            elif winner == -1:
                draws += 1

            # 重置棋盘
            self.game.board.initBoard(0)
            print("评估对局" +str(((i+2))/num_games) + "（后手）")
            winner, _ = self.game.playAIVsAI(self.rule_player, self.MCTSPlayer)
            if winner == 2:
                wins += 1
            elif winner == -1:
                draws += 1

            # 重置棋盘
            self.game.board.initBoard(0)

        win_rate = wins / num_games if num_games > 0 else 0
        print("评估结果：胜"+str(wins) + "局，平"+ str(draws) + "，胜率"+str(win_rate))
        return win_rate


if __name__ == '__main__':
    checkpoint_path = 'new/models2/checkpoint_150.pth'  # 续训：后面跑了10局后保存为checkpoint_10.pth
    metaZeta = MetaZeta(flag_is_train=True, load_from_checkpoint=checkpoint_path)

