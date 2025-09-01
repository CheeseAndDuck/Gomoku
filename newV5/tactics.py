import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import random
from tqdm import tqdm
from tkinter import END
import os
import csv
from datetime import datetime


class ResidualCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers={"filters": 38, "kernel_size": (3, 3)}, num_layers=20):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.filters = hidden_layers["filters"]
        k = hidden_layers["kernel_size"][0]
        pad = k // 2
        in_channels = input_dim[0]
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, self.filters, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm2d(self.filters),
            nn.LeakyReLU()
        )
        blocks = []
        for _ in range(self.num_layers):
            blocks.append(nn.Sequential(
                nn.Conv2d(self.filters, self.filters, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm2d(self.filters),
                nn.LeakyReLU(),
                nn.Conv2d(self.filters, self.filters, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm2d(self.filters)
            ))
        self.res_blocks = nn.ModuleList(blocks)
        h, w = input_dim[1], input_dim[2]

        self.policy_head = nn.Sequential(
            nn.Conv2d(self.filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU()
        )
        self.policy_fc = nn.Linear(2 * h * w, output_dim)

    def forward(self, x):
        out = self.conv_in(x)
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = F.leaky_relu(out + residual)

        p = self.policy_head(out)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        return policy_logits

    def BuildModel(self):
        return self

class PolicyValueNet(ResidualCNN):
    trainDataPoolSize = 18000 * 2
    trainBatchSize = 1024 * 2
    epochs = 10
    trainDataPool = deque(maxlen=trainDataPoolSize)
    kl_targ = 0.02
    learningRate = 2e-3
    LRfctor = 1.0
    training_log_path = "D:/Graduate_student_without_testing/985/10th/models2/tactic_training_log.csv"

    def __init__(self, input_dim):
        self.input_dim = input_dim
        ResidualCNN.__init__(self, input_dim=self.input_dim, output_dim=self.input_dim[1] * self.input_dim[2])
        self.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=self.learningRate)
        os.makedirs(os.path.dirname(self.training_log_path), exist_ok=True)
        
    def policy_NN(self, input):
        device = next(self.parameters()).device
        if isinstance(input, torch.Tensor):
            x = input.to(device)
        else:
            board_np = input.current_state().reshape(
                -1, self.input_dim[0], self.input_dim[1], self.input_dim[2]
            ).astype(np.float32)
            x = torch.from_numpy(board_np).to(device)

        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def fit_batch(self, batchBoard, batchProbs, learning_rate, optimizer=None):
        self.train()
        if optimizer is not None:
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        device = next(self.parameters()).device
        x = torch.from_numpy(np.array(batchBoard, dtype=np.float32)).to(device)
        target_probs = torch.from_numpy(np.array(batchProbs, dtype=np.float32)).to(device)

        logits = self(x)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(target_probs * log_probs).sum(dim=1).mean()
        loss = policy_loss

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return float(loss.detach().cpu().item())

    def save_training_log(self, avg_loss, avg_kl):
        try:
            file_exists = os.path.exists(self.training_log_path)
            with open(self.training_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'avg_loss', 'avg_kl', 'learning_rate_factor', 'batch_size', 'data_pool_size'])
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

    def update(self, scrollText=None, optimizer=None):
        print("开始训练策略网络")
        if len(self.trainDataPool) < self.trainBatchSize:
            if scrollText:
                scrollText.insert(END, f"策略网络数据不足: {len(self.trainDataPool)}/{self.trainBatchSize}\n")
                scrollText.see(END)
                scrollText.update()
            return None
        trainBatch = random.sample(self.trainDataPool, self.trainBatchSize)
        batchBoard = [data[0] for data in trainBatch]
        batchProbs = [data[1] for data in trainBatch]

        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x_old = torch.from_numpy(np.array(batchBoard, dtype=np.float32)).to(device)
            logits_old = self(x_old)
            batchProbsOld = F.softmax(logits_old, dim=1).cpu().numpy()

        losses = []
        for epoch in range(self.epochs):
            loss = self.fit_batch(batchBoard, batchProbs, self.learningRate * self.LRfctor, optimizer)
            losses.append(loss)
            
            # 计算KL散度，用于调整学习率
            with torch.no_grad():
                x_new = torch.from_numpy(np.array(batchBoard, dtype=np.float32)).to(device)
                logits_new = self(x_new)
                batchProbsNew = F.softmax(logits_new, dim=1).cpu().numpy()

            kl = np.mean(
                np.sum(batchProbsOld * (np.log(batchProbsOld + 1e-10) - np.log(batchProbsNew + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break
        avg_loss = sum(losses) / len(losses)
        
        # 根据KL散度调整学习率因子
        if kl > self.kl_targ * 2 and self.LRfctor > 0.1:
            self.LRfctor /= 1.5
        elif kl < self.kl_targ / 2 and self.LRfctor < 10:
            self.LRfctor *= 1.5
        scrollText.insert(END, '训练策略网络\n'+'loss:' + str(round(avg_loss, 4))+'KL:'+str(round(kl, 4))+'\n')
        scrollText.see(END)
        scrollText.update()
        self.save_training_log(avg_loss,kl)
        return avg_loss

    def memory(self, play_data):
        play_data = self.get_DataAugmentation(list(play_data)[:])
        self.trainDataPool.extend(play_data)

    def load_model(self, model_file):
        state = torch.load(model_file, map_location=next(self.parameters()).device)
        self.load_state_dict(state)

    def save_model(self, model_file):
        torch.save(self.state_dict(), model_file)

    def get_DataAugmentation(self, play_data):
        extendData = []
        for board, porbs, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_board = np.array([np.rot90(b, i) for b in board])
                equi_porbs = np.rot90(np.flipud(porbs.reshape(self.input_dim[1], self.input_dim[2])), i)
                extendData.append((equi_board, np.flipud(equi_porbs).flatten(), winner))

                equi_board = np.array([np.fliplr(s) for s in equi_board])
                equi_porbs = np.fliplr(equi_porbs)
                extendData.append((equi_board, np.flipud(equi_porbs).flatten(), winner))

        return extendData
