# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
from collections import deque
import numpy as np
import os

REPLAY_SIZE = 2000
small_BATCH_SIZE = 16
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01

class NET(nn.Module):
    def __init__(self, observation_height, observation_width, action_space):
        super(NET, self).__init__()
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(int((self.state_w/4) * (self.state_h/4) * 64), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.action_dim)
        
        self.relu = nn.ReLU()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.reshape(-1, int((self.state_w/4) * (self.state_h/4) * 64))
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class DQN(object):
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        
        self.eval_net = NET(observation_height, observation_width, action_space).to(self.device)
        self.target_net = NET(observation_height, observation_width, action_space).to(self.device)
        
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.epsilon = INITIAL_EPSILON
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.0005)
        self.loss_func = nn.MSELoss()
        
        # 模型保存路径
        self.model_file = model_file
        self.model_path = os.path.join(model_file, "pytorch_model.pth")
        self.log_file = log_file
        
        # 加载模型（如果存在）
        self._load_model()
        
        # 初始化目标网络
        self.update_target_network()
    
    def _load_model(self):
        """加载预训练模型"""
        if os.path.exists(self.model_path):
            print("模型存在，加载模型")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.epsilon = checkpoint.get('epsilon', INITIAL_EPSILON)
            print("模型加载完成")
        else:
            print("模型不存在，创建新模型")
            # 确保模型目录存在
            os.makedirs(self.model_file, exist_ok=True)
    
    def Choose_Action(self, state):
        """选择动作 - 兼容原接口名称"""
        return self.choose_action(state)
    
    def choose_action(self, state):
        """使用epsilon-greedy策略选择动作"""
        if random.random() <= self.epsilon:
            # 随机选择动作
            self.epsilon = max(FINAL_EPSILON, self.epsilon - (INITIAL_EPSILON - FINAL_EPSILON) / 10000)
            return random.randint(0, self.action_dim - 1)
        else:
            # 选择Q值最大的动作
            self.epsilon = max(FINAL_EPSILON, self.epsilon - (INITIAL_EPSILON - FINAL_EPSILON) / 10000)
            
            # 确保state是正确的tensor格式
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            
            # 添加batch维度如果需要
            if len(state.shape) == 3:  # [H, W, C] -> [1, C, H, W]
                state = state.unsqueeze(0)
            elif len(state.shape) == 2:  # [H, W] -> [1, 1, H, W]
                state = state.unsqueeze(0).unsqueeze(0)
            
            state = state.to(self.device)
            
            with torch.no_grad():
                q_values = self.eval_net(state)
                action = torch.argmax(q_values).item()
            
            return action
    
    def Store_Data(self, state, action, reward, next_state, done):
        """存储经验数据 - 兼容原接口名称"""
        self.store_data(state, action, reward, next_state, done)
    
    def store_data(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        # 创建one-hot编码的动作
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        
        # 存储经验
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
    
    def Train_Network(self, batch_size, num_step):
        """训练网络 - 兼容原接口名称"""
        self.train_network(batch_size, num_step)
    
    def train_network(self, batch_size, num_step):
        """训练网络"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 从经验回放缓冲区采样
        minibatch = random.sample(self.replay_buffer, batch_size)
        
        # 分离数据
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]
        
        # 转换为tensor
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        
        # 确保状态张量的形状正确 [batch, channel, height, width]
        if len(state_batch.shape) == 3:  # [batch, height, width] -> [batch, 1, height, width]
            state_batch = state_batch.unsqueeze(1)
        elif len(state_batch.shape) == 4 and state_batch.shape[3] == 1:  # [batch, height, width, 1] -> [batch, 1, height, width]
            state_batch = state_batch.permute(0, 3, 1, 2)
        
        if len(next_state_batch.shape) == 3:  # [batch, height, width] -> [batch, 1, height, width]
            next_state_batch = next_state_batch.unsqueeze(1)
        elif len(next_state_batch.shape) == 4 and next_state_batch.shape[3] == 1:  # [batch, height, width, 1] -> [batch, 1, height, width]
            next_state_batch = next_state_batch.permute(0, 3, 1, 2)
        
        # 计算当前Q值
        current_q_values = self.eval_net(state_batch)
        q_action = torch.sum(current_q_values * action_batch, dim=1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            
            target_q_values = reward_batch + GAMMA * max_next_q_values * (1 - torch.FloatTensor(done_batch).to(self.device))
        
        # 计算损失
        loss = self.loss_func(q_action, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 可选：打印训练信息
        if num_step % 100 == 0:
            print(f"Step {num_step}, Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")
    
    def Update_Target_Network(self):
        """更新目标网络 - 兼容原接口名称"""
        self.update_target_network()
    
    def update_target_network(self):
        """更新目标网络参数"""
        self.target_net.load_state_dict(self.eval_net.state_dict())
    
    def action(self, state):
        """测试时使用的动作选择（不使用epsilon-greedy）"""
        # 确保state是正确的tensor格式
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        # 添加batch维度如果需要
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        elif len(state.shape) == 2:
            state = state.unsqueeze(0).unsqueeze(0)
        
        state = state.to(self.device)
        
        with torch.no_grad():
            q_values = self.eval_net(state)
            action = torch.argmax(q_values).item()
        
        return action
    
    def save_model(self):
        """保存模型"""
        checkpoint = {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, os.path.join(self.model_file, "pytorch_save_model.pth"))
        print("模型已保存:pytorch_save_model.pth")
    
    def load_model(self, model_path=None):
        """加载指定路径的模型"""
        if model_path is None:
            model_path = self.model_path
        
        if os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', INITIAL_EPSILON)
            print("模型加载完成")
            return True
        else:
            print(f"模型文件不存在: {model_path}")
            return False
    
    def save_boss_victory_model(self, episode, episode_time, avg_reward):
        """击败boss时保存特殊模型和记录"""
        import datetime
        
        # 创建带回合数的模型文件名
        boss_model_filename = f"pytorch_model_episode_{episode}_boss_victory.pth"
        boss_model_path = os.path.join(self.model_file, boss_model_filename)
        
        # 保存模型
        checkpoint = {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'episode_time': episode_time,
            'avg_reward': avg_reward
        }
        torch.save(checkpoint, boss_model_path)
        
        # 创建同名txt记录文件
        txt_filename = f"pytorch_model_episode_{episode}_boss_victory.txt"
        txt_path = os.path.join(self.model_file, txt_filename)
        
        # 记录回合信息
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Boss击败记录\n")
            f.write(f"="*50 + "\n")
            f.write(f"回合数: {episode}\n")
            f.write(f"回合时间: {episode_time:.2f} 秒\n")
            f.write(f"回合平均分数: {avg_reward:.4f}\n")
            f.write(f"当前Epsilon值: {self.epsilon:.4f}\n")
            f.write(f"保存时间: {current_time}\n")
            f.write(f"模型文件: {boss_model_filename}\n")
        
        print(f"Boss击败！模型已保存到: {boss_model_path}")
        print(f"记录已保存到: {txt_path}")