import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RolloutPolicy(nn.Module):
    """AlphaGo 快速走子网络：轻量级CNN，用于MCTS模拟阶段快速推演至终局"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim  # (4, 15, 15)：4个特征层，15x15棋盘
        self.output_dim = output_dim  # 15*15=225：所有可能动作
        in_channels = input_dim[0]

        # 轻量CNN结构（1层卷积+全连接，平衡速度与性能）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # 全连接层：将卷积输出展平后映射到动作空间
        h, w = input_dim[1], input_dim[2]
        self.fc = nn.Linear(16 * h * w, output_dim)

    def forward(self, x):
        """前向传播：输入棋盘状态，输出动作概率"""
        out = self.conv(x)
        out = out.view(out.size(0), -1)  # 展平：(batch, 16*15*15)
        logits = self.fc(out)
        return logits  # 输出logits（后续用softmax转概率）

    def get_action(self, state):
        """根据当前棋盘状态，用快速走子网络选择动作（模拟阶段用）"""
        device = next(self.parameters()).device
        # 处理输入：Board对象 → numpy → torch张量
        if isinstance(state, np.ndarray):
            board_np = state.reshape(-1, self.input_dim[0], self.input_dim[1], self.input_dim[2]).astype(np.float32)
        else:
            board_np = state.current_state().reshape(-1, self.input_dim[0], self.input_dim[1], self.input_dim[2]).astype(np.float32)
        
        x = torch.from_numpy(board_np).to(device)
        self.eval()  # 推理模式，禁用Dropout/BatchNorm更新
        with torch.no_grad():  # 禁用梯度计算，提升速度
            logits = self(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # 转概率并返回CPU
        
        # 只选择可用动作（过滤已落子位置）
        avail_moves = state.getAvailableMoves()
        if not avail_moves:
            return None  # 棋盘满，返回空
        # 过滤无效动作的概率，重新归一化
        avail_probs = probs[avail_moves]
        avail_probs = avail_probs / np.sum(avail_probs)  # 数值稳定
        # 基于概率随机选择动作（模拟阶段的随机性）
        return np.random.choice(avail_moves, p=avail_probs)