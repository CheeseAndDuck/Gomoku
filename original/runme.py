from __future__ import print_function
import os
import tkinter as tk
import threading
from tkinter import *
from tkinter import scrolledtext
from tkinter import END
import random
import numpy as np
import pickle
from tqdm import tqdm
import math
import csv
import json
import torch
from torch import optim
import datetime

from MCTS import MCTS, TreeNode
from tactics import PolicyValueNet
from evaluateValue import ValueTrainer
from ruleAI import RuleBasedPlayer
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Board(object):
    """棋盘类（原有逻辑不变）"""
    def __init__(self, width=15, height=15, n_in_row=5):
        self.width = width
        self.height = height
        self.states = {}
        self.n_in_row = n_in_row
        self.players = [1, 2]

    def initBoard(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('板宽和板高不得小于{}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
        reversed_state = square_state[:, ::-1, :].copy()  # 添加.copy()消除负步长
        return reversed_state

    def do_move(self, move):
        # 添加防御性检查
        if move not in self.availables:
            raise ValueError(f"尝试落子无效位置 {move}，可用位置: {self.availables}")
        
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move


    def has_a_winner(self):
        width, height, states, n = self.width, self.height, self.states, self.n_in_row
        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < n * 2 - 1:
            return False, -1
        for m in moved:
            h, w = m // width, m % width
            player = states[m]
            if w in range(width - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n))) == 1:
                return True, player
            if h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1:
                return True, player
            if w in range(width - n + 1) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1:
                return True, player
            if w in range(n - 1, width) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1:
                return True, player
        return False, -1

    def gameIsOver(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def getCurrentPlayer(self):
        return self.current_player

    def getState(self):
        return self.current_state()

    def getAvailableMoves(self):
        return list(self.availables)


class Game():
    """游戏管理类"""
    boardWidth = 15
    boardHeight = 15
    n_in_row = 5
    flag_human_click = False
    move_human = -1

    def __init__(self, Canvas, scrollText, flag_is_shown=True, flag_is_train=True):
        self.flag_is_shown = flag_is_shown
        self.flag_is_train = flag_is_train
        self.board = Board(width=self.boardWidth, height=self.boardHeight, n_in_row=self.n_in_row)
        self.Canvas = Canvas
        self.scrollText = scrollText
        self.rect = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def Show(self, board, KEY=False):
        x = board.last_move // board.width
        y = board.last_move % board.height
        prev_player = board.players[0] if board.current_player == board.players[1] else board.players[1]
        self.drawPieces(player=prev_player, rc_pos=(x, y), Index=len(board.states))
        if KEY:
            playerName = 'you' if (self.flag_is_train == False and board.current_player != 1) else ('AI' if (self.flag_is_train == False) else f'AI-{board.current_player}')
            self.drawText(f"{len(board.states)} {playerName}: {x} {y}")

    def drawText(self, string):
        self.scrollText.insert(END, string + '\n')
        self.scrollText.see(END)
        self.scrollText.update()

    def drawPieces(self, player, rc_pos, Index, RADIUS=10, draw_rect=True):
        x, y = self.convert_rc_to_xy(rc_pos)
        if player == 1:
            OFFSET = RADIUS
            self.Canvas.create_line(x - OFFSET, y - OFFSET, x + OFFSET, y + OFFSET, width=2, fill='red')
            self.Canvas.create_line(x - OFFSET, y + OFFSET, x + OFFSET, y - OFFSET, width=2, fill='red')
        else:
            self.Canvas.create_oval(x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS, outline='blue', width=2)
        if draw_rect:
            if self.rect is None:
                OFFSET = 20
                self.rect = self.Canvas.create_rectangle(x - OFFSET, y - OFFSET, x + OFFSET, y + OFFSET, outline="#c1005d")
                self.rect_xy_pos = (x, y)
            else:
                new_x, new_y = self.convert_rc_to_xy(rc_pos)
                dx, dy = new_x - self.rect_xy_pos[0], new_y - self.rect_xy_pos[1]
                self.Canvas.move(self.rect, dx, dy)
                self.rect_xy_pos = (new_x, new_y)
        self.Canvas.update()

    def convert_rc_to_xy(self, rc_pos):
        SIDE = (435 - 400) / 2
        DELTA = (400 - 2) / self.boardWidth
        r, c = rc_pos
        x = c * DELTA + SIDE + DELTA / 2
        y = r * DELTA + SIDE + DELTA / 2
        return x, y

    def convert_xy_to_rc(self, xy_pos):
        SIDE = (435 - 400) / 2
        DELTA = (400 - 2) / self.boardWidth
        x, y = xy_pos
        r = int(math.floor((y - SIDE) / DELTA))
        c = int(math.floor((x - SIDE) / DELTA))
        r = max(0, min(self.boardWidth - 1, r))
        c = max(0, min(self.boardWidth - 1, c))
        return r, c

    def selfPlay(self, player, Index=0):
        self.board.initBoard()
        boards, probs, currentPlayer = [], [], []
        while True:
            # 新增：棋盘状态（numpy）转GPU张量
            board_state = torch.tensor(self.board.current_state(), dtype=torch.float32).unsqueeze(0).to(self.device)
            # 修改：传入GPU张量给getAction
            move, move_probs = player.getAction(self.board, self.flag_is_train, board_state)


            # move, move_probs = player.getAction(self.board, self.flag_is_train)
            boards.append(self.board.current_state())
            probs.append(move_probs)
            currentPlayer.append(self.board.current_player)
            self.board.do_move(move)
            if self.flag_is_shown:
                self.Show(self.board)
            gameOver, winner = self.board.gameIsOver()
            if gameOver:
                winners_z = np.zeros(len(currentPlayer))
                if winner != -1:
                    winners_z[np.array(currentPlayer) == winner] = 1.0
                    winners_z[np.array(currentPlayer) != winner] = -1.0
                player.resetMCTS()
                if self.flag_is_shown:
                    playerName = 'you' if (self.flag_is_train == False and self.board.current_player != 1) else ('AI' if (self.flag_is_train == False) else f'AI-{self.board.current_player}')
                    self.drawText(f"Game end. Winner is : {playerName}" if winner != -1 else "Game end. Tie")
                self.rect = None
                return winner, zip(boards, probs, winners_z)

    def humanMove(self, event):
        self.flag_human_click = True
        r, c = self.convert_xy_to_rc((event.x, event.y))
        self.move_human = r * self.boardWidth + c

    def playWithHuman(self, player):
        self.Canvas.bind("<Button-1>", self.humanMove)
        self.board.initBoard(0)
        KEY = 0
        while True:
            if self.board.current_player == 1:
                # 新增：棋盘状态转GPU张量
                board_state = torch.tensor(self.board.current_state(), dtype=torch.float32).unsqueeze(0).to(self.device)
                # 修改：传入GPU张量给getAction
                move, move_probs = player.getAction(self.board, self.flag_is_train, board_state)
            
                # move, move_probs = player.getAction(self.board, self.flag_is_train)
                self.board.do_move(move)
                KEY = 1
            else:
                if self.flag_human_click:
                    if self.move_human in self.board.availables:
                        self.flag_human_click = False
                        self.board.do_move(self.move_human)
                        KEY = 1
                    else:
                        self.flag_human_click = False
                        print("无效区域")
            if self.flag_is_shown and KEY == 1:
                self.Show(self.board)
                KEY = 0
            gameOver, winner = self.board.gameIsOver()
            if gameOver:
                player.resetMCTS()
                if self.flag_is_shown:
                    playerName = 'you' if (self.flag_is_train == False and self.board.current_player != 1) else ('AI' if (self.flag_is_train == False) else f'AI-{self.board.current_player}')
                    self.drawText(f"Game end. Winner is : {playerName}" if winner != -1 else "Game end. Tie")
                self.rect = None
                break

    def playAIVsAI(self, player_black, player_white, max_moves: int = 225):
        try:
            self.board.initBoard(0)
            boards, probs, currentPlayer = [], [], []
            for _ in range(max_moves):
                current_player_obj = player_black if self.board.current_player == 1 else player_white
                # 新增：棋盘状态转GPU张量
                board_state = torch.tensor(self.board.current_state(), dtype=torch.float32).unsqueeze(0).to(self.device)
                # 修改：传入GPU张量给getAction
                move, move_probs = current_player_obj.getAction(self.board, flag_is_train=False, board_state=board_state)
            
                # 检查移动是否有效
                if move not in self.board.availables:
                    print(f"警告: AI尝试无效移动 {move}，可用移动: {self.board.availables}")
                    # 选择第一个可用移动作为备选
                    if self.board.availables:
                        move = self.board.availables[0]
                    else:
                        break
                
                boards.append(self.board.current_state())
                probs.append(move_probs)
                currentPlayer.append(self.board.current_player)
                self.board.do_move(move)
                if self.flag_is_shown:
                    self.Show(self.board)
                gameOver, winner = self.board.gameIsOver()
                if gameOver:
                    winners_z = np.zeros(len(currentPlayer))
                    if winner != -1:
                        winners_z[np.array(currentPlayer) == winner] = 1.0
                        winners_z[np.array(currentPlayer) != winner] = -1.0
                    player_black.resetMCTS() if hasattr(player_black, 'resetMCTS') else None
                    player_white.resetMCTS() if hasattr(player_white, 'resetMCTS') else None
                    self.rect = None
                    return winner, zip(boards, probs, winners_z)
            player_black.resetMCTS() if hasattr(player_black, 'resetMCTS') else None
            player_white.resetMCTS() if hasattr(player_white, 'resetMCTS') else None
            self.rect = None
            return -1, zip(boards, probs, np.zeros(len(currentPlayer)))
        except Exception as e:
            print(f"AI对战中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return -1, []


class MCTSPlayer():
    """MCTS AI棋手"""
    def __init__(self, policy_net, value_net):
        self.simulations = 800
        self.factor = 2
        self.policy_net = policy_net
        self.value_net = value_net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.MCTS = MCTS(
            policy_NN=lambda s: self.policy_net.policy_NN(s),
            value_net=self.value_net,
            factor=self.factor,
            simulations=self.simulations
        )
    
    def resetMCTS(self):
        self.MCTS.updateMCTS(-1)
    
    def getAction(self, board, flag_is_train,board_state):
        emptySpacesBoard = board.availables
        move_probs = np.zeros(board.width * board.height)
        if len(emptySpacesBoard) > 0:
           
            # 获取落子位置和概率（此时probs应为NumPy数组）
            acts, probs = self.MCTS.getMoveProbs(board, flag_is_train, board_state)
            
            # 错误点：probs已经是NumPy数组，不需要调用.cpu()
            move_probs[list(acts)] = probs  # 直接使用probs



            # acts, probs = self.MCTS.getMoveProbs(board, flag_is_train)
            # move_probs[list(acts)] = probs
            if flag_is_train:
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                self.MCTS.updateMCTS(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.MCTS.updateMCTS(-1)
            return move, move_probs
        else:
            print("WARNING: the board is full")
    
    def update_reward_params(self, play_data, optimizer):
        self.MCTS.update_reward_params(play_data, optimizer)
    
    def __str__(self):
        return "MCTS {}".format(self.player)


class MetaZeta(threading.Thread):
    save_ParaFreq = 2  # 每2局保存一次进度
    MAX_Games = 110  # 总训练局数（可通过加载checkpoint修改）

    def __init__(self, flag_is_shown=True, flag_is_train=True):
        super(MetaZeta, self).__init__()
        self.flag_is_shown = flag_is_shown
        self.flag_is_train = flag_is_train

        # 初始化GPU设
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if not torch.cuda.is_available():
        #     self.drawScrollText("未检测到GPU，使用CPU运行")

        # UI初始化
        self.window = tk.Tk()
        self.window.resizable(0, 0)
        self.window.title('Cheese')
        self.window.geometry('810x500')

        # 原有按钮：开始、重置
        self.btStart = tk.Button(self.window, text='开始训练/对战', command=lambda: self.thredaTrain(self.start_process))
        self.btStart.place(x=480, y=10)
        self.btReset = tk.Button(self.window, text='重置棋盘', command=self.resetCanvas)
        self.btReset.place(x=580, y=10)

        # 新增：加载训练进度相关UI
        self.load_label = tk.Label(self.window, text='Checkpoint路径：')
        self.load_label.place(x=480, y=40)
        self.load_entry = tk.Entry(self.window, width=35)
        self.load_entry.place(x=480, y=65)
        self.load_entry.insert(0, "checkpoints/checkpoint_gameX.pth")  # 默认提示
        self.btLoad = tk.Button(self.window, text='加载训练进度', command=self.load_checkpoint_from_entry)
        self.btLoad.place(x=480, y=95)

        # 模式选择
        self.iv_default = IntVar()
        self.rb_default1 = Radiobutton(self.window, text='自我对弈训练', value=1, variable=self.iv_default)
        self.rb_default2 = Radiobutton(self.window, text='人机对战', value=2, variable=self.iv_default)
        self.rb_default1.place(x=480, y=130)
        self.rb_default2.place(x=620, y=130)
        self.iv_default.set(2)

        # 棋盘与日志区域
        self.canvas = tk.Canvas(self.window, bg='white', height=435, width=435)
        self.scrollText = scrolledtext.ScrolledText(self.window, width=38, height=24)

        # 核心组件初始化
        self.game = Game(Canvas=self.canvas, scrollText=self.scrollText, flag_is_shown=self.flag_is_shown, flag_is_train=self.flag_is_train)
        self.policy_net = PolicyValueNet((4, self.game.boardWidth, self.game.boardHeight))
        self.value_net = ValueTrainer((4, self.game.boardWidth, self.game.boardHeight))
        self.MCTSPlayer = MCTSPlayer(policy_net=self.policy_net, value_net=self.value_net)
        self.rule_player = RuleBasedPlayer()

        # 训练进度相关变量
        self.current_loaded_game = 0  # 加载后已完成的局数
        self.tacticLoss = []          # 策略网络损失记录
        self.valueLoss = []           # 价值网络损失记录
        self.loaded_checkpoint = None # 加载的进度数据
        self.eval_count = 0           # 评估计数器

        # 设备配置 - 只对策略网络应用设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net.to(self.device)
        # 移除对ValueTrainer的to()方法调用，因为它不是nn.Module
        
        # 绘制棋盘与日志区域
        self.DrawCanvas((30, 30))
        self.DrawText((480, 160))
        self.DrawRowsCols((42, 470), (10, 35))

        self.window.mainloop()


    def save_training_log(self, avg_loss, avg_kl):
        """保存训练记录，包括α值"""
        try:
            file_exists = os.path.exists(self.training_log_path)
            
            with open(self.training_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'avg_loss', 'avg_kl', 'alpha', 'learning_rate_factor', 'batch_size', 'data_pool_size'])
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    avg_loss,
                    avg_kl,
                    self.LRfctor,
                    self.trainBatchSize,
                    len(self.trainDataPool)
                ])
            
            print(f"训练记录已保存到 {self.training_log_path}")
        
        except Exception as e:
            print(f"保存训练记录失败: {e}")



    def save_training_checkpoint(self, current_game, optimizer, save_path="checkpoints"):
        """保存完整训练进度（不再包含奖励参数）"""
        os.makedirs(save_path, exist_ok=True)
        checkpoint_path = os.path.join(save_path, f"checkpoint_game{current_game}.pth")

        # 收集所有需保存的状态
        checkpoint = {
            # 1. 训练进度
            "current_game": current_game,
            "max_games": self.MAX_Games,
            "tactic_loss": self.tacticLoss,
            "value_loss": self.valueLoss,

            # 2. 网络参数
            "policy_net_state_dict": self.policy_net.state_dict(),

            # 3. 优化器状态（只保存策略网络的）
            "optimizer_state_dict": optimizer.state_dict(),

            # 4. 数据池
            "policy_train_data_pool": self.policy_net.trainDataPool,
            "value_train_data_pool": self.value_net.trainDataPool
        }

        # 尝试保存价值网络（如果它有state_dict方法）
        if hasattr(self.value_net, 'state_dict'):
            try:
                checkpoint["value_net_state_dict"] = self.value_net.state_dict()
            except Exception as e:
                self.drawScrollText(f"价值网络状态保存警告：{str(e)}")

        # 保存文件
        try:
            torch.save(checkpoint, checkpoint_path)
            self.drawScrollText(f"训练进度已保存至：{checkpoint_path}")
        except Exception as e:
            self.drawScrollText(f"保存进度失败：{str(e)}")



    def load_training_checkpoint(self, checkpoint_path):
        """加载训练进度，恢复所有关键状态（不再恢复奖励参数）"""
        if not os.path.exists(checkpoint_path):
            self.drawScrollText(f"文件不存在：{checkpoint_path}")
            return None

        try:
            # 加载文件（自动适配CPU/GPU）
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 1. 恢复训练进度
            self.current_loaded_game = checkpoint["current_game"]
            # self.MAX_Games = checkpoint.get("max_games", self.MAX_Games)
            self.tacticLoss = checkpoint.get("tactic_loss", [])
            self.valueLoss = checkpoint.get("value_loss", [])

            # 2. 恢复网络参数
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            
            # 尝试恢复价值网络（如果存在且支持）
            if "value_net_state_dict" in checkpoint and hasattr(self.value_net, 'load_state_dict'):
                try:
                    self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
                    self.drawScrollText("已恢复价值网络参数")
                except Exception as e:
                    self.drawScrollText(f"价值网络参数恢复警告：{str(e)}")

            self.drawScrollText(f"已恢复策略网络参数（完成{self.current_loaded_game}/{self.MAX_Games}局）")

            # 3. 恢复数据池
            self.policy_net.trainDataPool = checkpoint["policy_train_data_pool"]
            self.value_net.trainDataPool = checkpoint["value_train_data_pool"]
            self.drawScrollText(f"已恢复数据池（策略池：{len(self.policy_net.trainDataPool)}条，价值池：{len(self.value_net.trainDataPool)}条）")

            self.loaded_checkpoint = checkpoint
            return checkpoint

        except Exception as e:
            self.drawScrollText(f"加载进度失败：{str(e)}")
            return None

    def load_checkpoint_from_entry(self):
        checkpoint_path = self.load_entry.get().strip()
        if not checkpoint_path:
            self.drawScrollText("❌ 请输入Checkpoint文件路径")
            return
        self.load_training_checkpoint(checkpoint_path)

    def start_process(self):
        if self.iv_default.get() == 1:
            self.flag_is_train = True
            self.game.flag_is_train = True
            self.train()  # 启动训练（支持续训）
        else:
            self.flag_is_train = False
            self.game.flag_is_train = False
            self.battle_mode()  # 启动对战

    def train(self):
        # 确定起始局数
        start_game = self.current_loaded_game if self.loaded_checkpoint is not None else 0
        if start_game > 0:
            self.drawScrollText(f"从第{start_game + 1}局开始续训（总局数：{self.MAX_Games}）")

        # 创建优化器 - 只包含策略网络参数，不再包含奖励参数
        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_net.learningRate * self.policy_net.LRfctor)

        # 恢复优化器状态（如果存在）
        if self.loaded_checkpoint is not None:
            try:
                # 只恢复策略网络的优化器状态
                policy_optimizer_state = {}
                checkpoint_optimizer_state = self.loaded_checkpoint["optimizer_state_dict"]
                
                # 筛选出只属于策略网络的参数
                for key in checkpoint_optimizer_state:
                    if key.startswith('policy_net.'):
                        policy_optimizer_state[key] = checkpoint_optimizer_state[key]
                
                if policy_optimizer_state:
                    optimizer.load_state_dict(policy_optimizer_state)
                    self.drawScrollText("已恢复策略网络优化器状态")
                else:
                    self.drawScrollText("无策略网络优化器状态可恢复")
            except Exception as e:
                self.drawScrollText(f"恢复优化器失败，使用新优化器：{str(e)}")

        # 训练循环
        for oneGame in range(start_game, self.MAX_Games):
            current_game_num = oneGame + 1
            if self.flag_is_train:
                self.drawScrollText(f"\n正在进行第{current_game_num}局自我对弈...")
                winner, play_data = self.game.selfPlay(self.MCTSPlayer, Index=current_game_num)
                play_data = list(play_data)

                # 存储训练数据
                self.policy_net.memory(play_data)
                self.value_net.memory(play_data)

                # 保存单局对弈数据到CSV
                try:
                    self.save_to_csv(play_data, current_game_num, winner)
                    self.drawScrollText(f"单局数据已保存至 play_chess.csv")
                except Exception as e:
                    self.drawScrollText(f"保存单局数据失败：{str(e)}")

                # 更新网络参数（不再更新奖励参数）
                tactic_loss = self.policy_net.update(scrollText=self.scrollText, optimizer=optimizer)
                value_loss = self.value_net.update(scrollText=self.scrollText)
                if tactic_loss is not None:
                    self.tacticLoss.append(tactic_loss)
                if value_loss is not None:
                    self.valueLoss.append(value_loss)
                    
                # 根据价值网络损失更新α
                # if value_loss is not None:
                #     self.MCTSPlayer.MCTS.update_alpha(performance_metric=value_loss)
                    
                # # 显示当前α值
                # self.drawScrollText(f"当前α值: {self.MCTSPlayer.MCTS.alpha:.3f} " +
                #                 f"(价值网络权重: {self.MCTSPlayer.MCTS.alpha:.1%}, " +
                #                 f"累积奖惩权重: {1 - self.MCTSPlayer.MCTS.alpha:.1%})")

                # 显示数据池收集进度
                if len(self.policy_net.trainDataPool) <= self.policy_net.trainBatchSize:
                    policy_progress = len(self.policy_net.trainDataPool) / self.policy_net.trainBatchSize * 100
                    value_progress = len(self.value_net.trainDataPool) / self.value_net.trainBatchSize * 100
                    self.drawScrollText(f"数据池进度：策略网络 {policy_progress:.1f}%，价值网络 {value_progress:.1f}%")

                # 每save_ParaFreq局保存一次完整进度
                if current_game_num % self.save_ParaFreq == 0:
                    self.policy_net.save_model(f'D:/Graduate_student_without_testing/985/10th-rightone/models2/{current_game_num}tactic.model')
                    
                    # 尝试保存价值网络（如果支持）
                    if hasattr(self.value_net, 'save_model'):
                        try:
                            self.value_net.save_model(f'D:/Graduate_student_without_testing/985/10th-rightone/models2/{current_game_num}evaluateValue.model')
                        except Exception as e:
                            self.drawScrollText(f"价值网络模型保存警告：{str(e)}")
                    
                    # 保存训练检查点（不再包含奖励参数）
                    self.save_training_checkpoint(current_game=current_game_num, optimizer=optimizer)

                # 重置棋盘显示
                self.canvas.delete("all")
                self.DrawCanvas((30, 30))
            else:
                # 对战模式（加载模型）
                if self.loaded_checkpoint is not None:
                    self.drawScrollText("从加载的进度中使用对战模型")
                else:
                    try:
                        model_path = f'D:/Graduate_student_without_testing/985/10th-rightone/models2/{current_game_num}tactic.model'
                        self.policy_net.load_model(model_path)
                        
                        # 尝试加载价值网络（如果支持）
                        value_path = f'D:/Graduate_student_without_testing/985/10th-rightone/models2/{current_game_num}evaluateValue.model'
                        if hasattr(self.value_net, 'load_model'):
                            self.value_net.load_model(value_path)
                            
                        self.drawScrollText(f"加载对战模型：{model_path}")
                    except Exception as e:
                        self.drawScrollText(f"加载对战模型失败：{str(e)}")
                self.game.playWithHuman(self.MCTSPlayer)
                return

        # 训练结束：保存最终进度+评估
        self.drawScrollText("训练结束！开始最终评估...")
        final_win_rate = self.evaluate(num_games=20)
        self.drawScrollText(f"最终评估结果：胜率 {final_win_rate:.2%}")

        # 保存最终模型与进度
        self.policy_net.save_model("D:/Graduate_student_without_testing/985/10th-rightone/models2/final_tactic.model")
        
        # 尝试保存最终价值网络（如果支持）
        if hasattr(self.value_net, 'save_model'):
            try:
                self.value_net.save_model("D:/Graduate_student_without_testing/985/10th-rightone/models2/final_evaluateValue.model")
            except Exception as e:
                self.drawScrollText(f"最终价值网络模型保存警告：{str(e)}")
                
        self.save_training_checkpoint(current_game=self.MAX_Games, optimizer=optimizer, save_path="checkpoints")
        self.drawScrollText("已保存最终模型与训练进度！")


    # def update_alpha_based_on_performance(self, win_rate=None, value_loss=None):
    #     """
    #     基于性能指标动态调整α
    #     win_rate: 近期胜率
    #     value_loss: 价值网络近期损失
    #     """
    #     # 默认增长因子
    #     growth_factor = 1.0
    #
    #     # 基于胜率调整
    #     if win_rate is not None:
    #         if win_rate > 0.7:  # 高胜率，增加价值网络权重
    #             growth_factor *= 1.5
    #         elif win_rate < 0.3:  # 低胜率，减缓增加
    #             growth_factor *= 0.7
    #
    #     # 基于价值网络损失调整
    #     if value_loss is not None:
    #         if value_loss < 0.1:  # 低损失，增加价值网络权重
    #             growth_factor *= 1.5
    #         elif value_loss > 0.3:  # 高损失，减缓增加
    #             growth_factor *= 0.7
    #
    #     # 更新α值
    #     self.alpha = min(self.max_alpha, self.alpha + self.alpha_growth_rate * growth_factor)
    #
    #     # 记录α值变化
    #     self.alpha_history.append(self.alpha)


    def battle_mode(self):
        try:
            self.policy_net.load_model("models/Version1.model")
            
            # 尝试加载价值网络（如果支持）
            if hasattr(self.value_net, 'load_model'):
                try:
                    self.value_net.load_model("models/Version1_value.model")
                except Exception as e:
                    self.drawScrollText(f"加载价值网络模型警告：{str(e)}")
                    
            self.drawScrollText("已加载训练好的模型")
        except Exception as e:
            self.drawScrollText(f"加载模型失败：{str(e)}")
        num_battles = 10
        self.drawScrollText(f"开始与规则AI对战（共{num_battles}局）...")
        win_rate = self.evaluate(num_battles)
        self.drawScrollText(f"对战结束：胜率 {win_rate:.2%}")

    def thredaTrain(self, func):
        myThread = threading.Thread(target=func)
        myThread.daemon = True
        myThread.start()

    def save_to_csv(self, play_data, game_index, winner, filepath='D:/Graduate_student_without_testing/985/10th-rightone/play_chess.csv'):
        seq = list(play_data)
        def serialize(arr): return json.dumps(np.array(arr).tolist(), ensure_ascii=False)
        rows = []
        final_state = self.game.board.current_state()
        if winner == -1:
            idx_first = max([i for i in range(len(seq)) if i % 2 == 0], default=-1)
            idx_second = max([i for i in range(len(seq)) if i % 2 == 1], default=-1)
            if idx_first != -1:
                _, probs_first, _ = seq[idx_first]
                rows.append({"game": game_index, "result": 0, "state": serialize(final_state), "move_probs": serialize(probs_first)})
            if idx_second != -1:
                _, probs_second, _ = seq[idx_second]
                rows.append({"game": game_index, "result": 0, "state": serialize(final_state), "move_probs": serialize(probs_second)})
        else:
            idx_win = None
            idx_lose = None
            for i, (_, _, wz) in enumerate(seq):
                if wz == 1:
                    idx_win = i
                elif wz == -1:
                    idx_lose = i
            if idx_win is not None:
                _, probs_w, _ = seq[idx_win]
                rows.append({"game": game_index, "result": 1, "state": serialize(final_state), "move_probs": serialize(probs_w)})
            if idx_lose is not None:
                _, probs_l, _ = seq[idx_lose]
                rows.append({"game": game_index, "result": -1, "state": serialize(final_state), "move_probs": serialize(probs_l)})
        file_exists = os.path.exists(filepath)
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['game', 'result', 'state', 'move_probs'])
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def DrawCanvas(self, canvas_pos):
        x, y = canvas_pos
        SIDE = (435 - 400) / 2
        for i in range(self.game.boardWidth + 1):
            pos = i * (400 - 2) / self.game.boardWidth
            self.canvas.create_line(SIDE, SIDE + pos, SIDE + 400, SIDE + pos)
            self.canvas.create_line(SIDE + pos, SIDE, SIDE + pos, SIDE + 400)
        self.canvas.place(x=x, y=y)

    def DrawRowsCols(self, rspos, cspos):
        rx, ry = rspos
        cx, cy = cspos
        DELTA = (400 - 2) / self.game.boardWidth
        SIDE = (435 - 400) / 2
        for i in range(self.game.boardWidth):
            tk.Label(self.window, text=str(i)).place(x=cx, y=cy + i * DELTA + DELTA / 2)
            tk.Label(self.window, text=str(i)).place(x=rx + i * DELTA + DELTA / 2, y=ry)

    def DrawText(self, xy_pos):
        x, y = xy_pos
        self.scrollText.place(x=x, y=y)

    def drawScrollText(self, string):
        self.scrollText.insert(END, string + '\n')
        self.scrollText.see(END)
        self.scrollText.update()

    def resetCanvas(self):
        self.canvas.delete("all")
        self.scrollText.delete(1.0, END)
        self.DrawCanvas((30, 30))

    def evaluate(self, num_games=10):
        """评估函数，同时更新α值"""
        original_show_flag = self.game.flag_is_shown
        self.game.flag_is_shown = True # 评估时不显示
        
        wins, draws = 0, 0
        for i in range(0, num_games, 2):
            # 先手对局
            self.resetCanvas()
            winner, _ = self.game.playAIVsAI(self.MCTSPlayer, self.rule_player)
            wins += 1 if winner == 1 else 0
            draws += 1 if winner == -1 else 0
            self.game.board.initBoard(0)
            self.MCTSPlayer.resetMCTS()

            # 后手对局
            if i + 2 > num_games:
                break
            self.resetCanvas()
            winner, _ = self.game.playAIVsAI(self.rule_player, self.MCTSPlayer)
            wins += 1 if winner == 2 else 0
            draws += 1 if winner == -1 else 0
            self.game.board.initBoard(0)
            self.MCTSPlayer.resetMCTS()

        self.game.flag_is_shown = original_show_flag
        win_rate = wins / num_games if num_games > 0 else 0
        
        # 根据胜率更新α - 修复：使用正确的参数名
        # self.MCTSPlayer.MCTS.update_alpha(performance_metric=win_rate)

        self.drawScrollText(f"评估汇总：{wins}胜/{draws}平/{num_games - wins - draws}负")
        self.drawScrollText(f"当前胜率: {win_rate:.2%}")
        
        return win_rate


if __name__ == '__main__':
    metaZeta = MetaZeta()


    """
    该代码时没有使用累积折扣奖惩的
    """