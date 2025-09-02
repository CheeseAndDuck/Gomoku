import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import os
import csv
from datetime import datetime
import random

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers={"filters": 38, "kernel_size": (3, 3)}, num_layers=5):
        super().__init__()
        self.input_dim = input_dim
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
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(self.filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.value_fc1 = nn.Linear(1 * h * w, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = F.leaky_relu(out + residual)

        v = self.value_head(out)
        v = v.view(v.size(0), -1)
        v = F.leaky_relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(1)
        return v



class ValueTrainer:
    def __init__(self, input_dim, lr=1e-3, pool_size=10000, batch_size=512, epochs=5):
        self.input_dim = input_dim
        self.value_net = ValueNetwork(input_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.value_net.to(self.device)
        
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.lr = lr
        self.trainDataPool = deque(maxlen=pool_size)
        self.trainBatchSize = batch_size
        self.epochs = epochs
        self.training_log_path = "D:/Graduate_student_without_testing/985/10th/models2/evaluateValue_training_log.csv"
        os.makedirs(os.path.dirname(self.training_log_path), exist_ok=True)
        

    def evaluate(self, input):
        if isinstance(input, torch.Tensor):
            x = input.to(self.device)
        else:
            board_np = input.current_state().reshape(
                -1, self.input_dim[0], self.input_dim[1], self.input_dim[2]
            ).astype(np.float32)
            x = torch.from_numpy(board_np).to(self.device)

        self.value_net.eval()
        with torch.no_grad():
            value = self.value_net(x)
        return value.cpu().numpy()
    
    def memory(self, play_data):
        value_data = [(state, winner) for state, _, winner in play_data]
        self.trainDataPool.extend(value_data)
    
    def fit_batch(self, batchBoard, batchWinner):
        self.value_net.train()
        x = torch.from_numpy(np.array(batchBoard, dtype=np.float32)).to(self.device)
        target_values = torch.from_numpy(np.array(batchWinner, dtype=np.float32)).to(self.device)

        values = self.value_net(x)
        value_loss = F.mse_loss(values, target_values)
        
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        
        return float(value_loss.detach().cpu().item())
    


    def save_training_log(self, avg_loss):
        try:
            log_dir = os.path.dirname(self.training_log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
       
            data = {
                'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'avg_loss': [avg_loss],
                'batch_size': [self.trainBatchSize],
                'data_pool_size': [len(self.trainDataPool)]
            }

            file_exists = os.path.exists(self.training_log_path)
            
            with open(self.training_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(data)
            print(f"训练记录已保存到 {self.training_log_path}")
        except Exception as e:
            print(f"保存训练记录失败: {e}")



    def update(self):
        if len(self.trainDataPool) < self.trainBatchSize:
            return None
        
        batch_data = random.sample(self.trainDataPool, self.trainBatchSize)
        batchBoard = [data[0] for data in batch_data]
        batchWinner = [data[1] for data in batch_data]
        
        losses = []
        for epoch in range(self.epochs):
            loss = self.fit_batch(batchBoard, batchWinner)
            losses.append(loss)
        avg_loss = sum(losses) / len(losses)
        print( '训练评估网络\n'+'loss:' + str(round(avg_loss, 4))+'\n')
        self.save_training_log(avg_loss)
        return avg_loss
    
    def load_model(self, model_file):
        state_dict = torch.load(model_file, map_location=self.device)
        self.value_net.load_state_dict(state_dict)
        print(f"Loaded value model from {model_file}")
    
    def save_model(self, model_file):
        torch.save(self.value_net.state_dict(), model_file)
        print(f"Saved value model to {model_file}")
