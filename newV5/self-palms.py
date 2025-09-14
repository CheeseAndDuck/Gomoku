from __future__ import print_function
import os
import tkinter as tk
import threading
from tkinter import *
from tkinter import scrolledtext
from tkinter import END, filedialog
import random
import numpy as np
import pickle
from tqdm import tqdm
import math
import csv
import json
import torch
from torch import optim
import warnings

# 忽略特定的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

# 假设这些模块已经存在
from MCTS import MCTS, TreeNode
from tactics import PolicyValueNet
from evaluateValue import ValueTrainer, ValueNetwork
from ruleAI import RuleBasedPlayer
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Board(object):
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
        reversed_state = square_state[:, ::-1, :].copy()
        return reversed_state

    def do_move(self, move):
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
            playerName = 'AI-1' if prev_player == 1 else 'AI-2'
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

    def playAIVsAI(self, player_black, player_white, max_moves: int = 225):
        try:
            self.board.initBoard(0)
            boards, probs, currentPlayer = [], [], []
            for _ in range(max_moves):
                current_player_obj = player_black if self.board.current_player == 1 else player_white
                board_state = torch.tensor(self.board.current_state(), dtype=torch.float32).unsqueeze(0).to(self.device)
                move, move_probs = current_player_obj.getAction(self.board, flag_is_train=False, board_state=board_state)
                if move not in self.board.availables:
                    print(f"警告: AI尝试无效移动 {move}，可用移动: {self.board.availables}")
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
    def __init__(self, policy_net, value_net, player_name="MCTS Player"):
        self.simulations = 800
        self.factor = 2
        self.policy_net = policy_net
        self.value_net = value_net
        self.player_name = player_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.MCTS = MCTS(
            policy_NN=lambda s: self.policy_net.policy_NN(s),
            value_net=self.value_net,
            factor=self.factor,
            simulations=self.simulations
        )
    
    def resetMCTS(self):
        self.MCTS.updateMCTS(-1)
    
    def getAction(self, board, flag_is_train, board_state):
        emptySpacesBoard = board.availables
        move_probs = np.zeros(board.width * board.height)
        if len(emptySpacesBoard) > 0:
            acts, probs = self.MCTS.getMoveProbs(board, flag_is_train, board_state)
            move_probs[list(acts)] = probs
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
        return self.player_name


class PTHAIVsAIVisualizer(threading.Thread):
    def __init__(self, flag_is_shown=True):
        super(PTHAIVsAIVisualizer, self).__init__()
        self.flag_is_shown = flag_is_shown

        # 初始化GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("未检测到GPU，使用CPU运行")

        # UI初始化
        self.window = tk.Tk()
        self.window.resizable(0, 0)
        self.window.title('五子棋对弈')
        self.window.geometry('810x500')

        # 模型选择区域
        self.model1_label = tk.Label(self.window, text='AI1 PTH文件路径：')
        self.model1_label.place(x=480, y=10)
        self.model1_entry = tk.Entry(self.window, width=35)
        self.model1_entry.place(x=480, y=35)
        self.model1_entry.insert(0, "checkpoints/checkpoint_gameX.pth")  # 默认提示
        self.model1_btn = tk.Button(self.window, text='浏览', command=lambda: self.browse_file(self.model1_entry))
        self.model1_btn.place(x=750, y=33)

        self.model2_label = tk.Label(self.window, text='AI2 PTH文件路径：')
        self.model2_label.place(x=480, y=65)
        self.model2_entry = tk.Entry(self.window, width=35)
        self.model2_entry.place(x=480, y=90)
        self.model2_entry.insert(0, "checkpoints/checkpoint_gameY.pth")  # 默认提示
        self.model2_btn = tk.Button(self.window, text='浏览', command=lambda: self.browse_file(self.model2_entry))
        self.model2_btn.place(x=750, y=88)

        # 按钮
        self.btLoad = tk.Button(self.window, text='加载模型并开始对弈', command=self.start_ai_vs_ai)
        self.btLoad.place(x=480, y=120)
        self.btReset = tk.Button(self.window, text='重置棋盘', command=self.resetCanvas)
        self.btReset.place(x=620, y=120)

        # 棋盘与日志区域
        self.canvas = tk.Canvas(self.window, bg='white', height=435, width=435)
        self.scrollText = scrolledtext.ScrolledText(self.window, width=38, height=24)

        # 核心组件初始化
        self.game = Game(Canvas=self.canvas, scrollText=self.scrollText, flag_is_shown=self.flag_is_shown, flag_is_train=False)
        
        # 绘制棋盘与日志区域
        self.DrawCanvas((30, 30))
        self.DrawText((480, 150))
        self.DrawRowsCols((42, 470), (10, 35))

        self.window.mainloop()

    def browse_file(self, entry_widget):
        filename = filedialog.askopenfilename(
            title="选择PTH模型文件",
            filetypes=[("PTH files", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            entry_widget.delete(0, END)
            entry_widget.insert(0, filename)

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

    def load_pth_model(self, pth_path):
        """从PTH文件加载模型"""
        try:
            # 加载检查点
            checkpoint = torch.load(pth_path, map_location=self.device)
            
            # 创建策略网络
            policy_net = PolicyValueNet((4, self.game.boardWidth, self.game.boardHeight))
            
            # 加载策略网络状态
            if "policy_net_state_dict" in checkpoint:
                policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
                self.drawScrollText(f"已加载策略网络参数从: {pth_path}")
            else:
                self.drawScrollText(f"警告: PTH文件中未找到策略网络参数")
                return None, None
            
            # 创建价值网络
            value_net = ValueTrainer((4, self.game.boardWidth, self.game.boardHeight))
            
            # 加载价值网络状态（如果存在）
            if "value_net_state_dict" in checkpoint and hasattr(value_net, 'load_state_dict'):
                try:
                    value_net.load_state_dict(checkpoint["value_net_state_dict"])
                    self.drawScrollText(f"已加载价值网络参数从: {pth_path}")
                except Exception as e:
                    self.drawScrollText(f"加载价值网络参数警告: {str(e)}")
            else:
                self.drawScrollText(f"PTH文件中未找到价值网络参数，使用默认价值网络")
            
            # 显示训练进度信息
            if "current_game" in checkpoint:
                self.drawScrollText(f"模型训练进度: {checkpoint['current_game']}局")
            
            return policy_net, value_net
        except Exception as e:
            self.drawScrollText(f"加载PTH模型失败: {str(e)}")
            return None, None

    def start_ai_vs_ai(self):
        """开始AI对AI对弈"""
        # 获取模型路径
        model1_path = self.model1_entry.get().strip()
        model2_path = self.model2_entry.get().strip()
        
        if not model1_path or not model2_path:
            self.drawScrollText("请选择两个PTH模型文件")
            return
        
        # 加载模型
        self.drawScrollText("正在加载AI1模型...")
        policy_net1, value_net1 = self.load_pth_model(model1_path)
        if policy_net1 is None:
            self.drawScrollText("AI1模型加载失败，无法开始对弈")
            return
        
        self.drawScrollText("正在加载AI2模型...")
        policy_net2, value_net2 = self.load_pth_model(model2_path)
        if policy_net2 is None:
            self.drawScrollText("AI2模型加载失败，无法开始对弈")
            return
        
        # 创建MCTS玩家
        ai1_player = MCTSPlayer(policy_net1, value_net1, "AI-1")
        ai2_player = MCTSPlayer(policy_net2, value_net2, "AI-2")
        
        # 开始对弈线程
        threading.Thread(target=self.run_ai_vs_ai, args=(ai1_player, ai2_player), daemon=True).start()

    def run_ai_vs_ai(self, ai1_player, ai2_player):
        """运行AI对AI对弈"""
        self.drawScrollText("开始AI对AI对弈...")
        self.resetCanvas()
        
        # 进行对弈
        winner, _ = self.game.playAIVsAI(ai1_player, ai2_player)
        
        # 显示结果
        if winner == 1:
            self.drawScrollText("对弈结束: AI-1 (黑子) 获胜!")
        elif winner == 2:
            self.drawScrollText("对弈结束: AI-2 (白子) 获胜!")
        else:
            self.drawScrollText("对弈结束: 平局!")


if __name__ == '__main__':
    visualizer = PTHAIVsAIVisualizer()