# 🎮 DQN 埃尔登法环 AI 训练项目

一个基于深度Q网络(DQN)的埃尔登法环游戏AI训练项目，使用PyTorch实现，能够自主学习游戏策略并与Boss战斗。

## 谢鸣
感谢以下项目为本项目提供的基本思路：
https://github.com/analoganddigital/DQN_play_sekiro 
https://github.com/XR-stb/DQN_WUKONG 

### 安装步骤

#### 克隆项目
```bash
git clone https://github.com/hhhyxnh/DQN_eldenring.git
cd DQN_eldenring
```

#### 创建虚拟环境
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

#### 安装依赖
```bash
pip install -r requirements.txt
```

## 📁 结构

```
DQN_eldenring/
├── DQN_eldenring_training_pytorch.py  # 主训练脚本
├── DQN_eldenring_test_pytorch.py      # 测试脚本
├── DQN_pytorch_gpu.py                 # DQN模型实现
├── directkeys.py                      # 按键控制模块
├── getkeys.py                         # 按键检测模块
├── restart.py                         # 重启控制模块
├── requirements.txt                   # 依赖列表
├── logo.png                          # 游戏窗口识别图标
├── model_gpu/                        # 模型保存目录
│   ├── pytorch_model.pth             # 主模型文件
│   └── pytorch_model_episode_*.pth   # 特定回合模型
└── venv/                             # 虚拟环境
```

## 模型配置


## 游戏设置


## 免责声明

本项目仅用于学习和研究目的，请遵守游戏服务条款和相关法律法规。


### 🧠 AI核心功能
- **深度Q网络(DQN)**：使用PyTorch实现的神经网络模型
- **经验回放**：存储和重放历史经验以提高学习效率
- **目标网络**：稳定训练过程的双网络结构
- **ε-贪婪策略**：平衡探索与利用的动作选择机制

### 🎯 游戏交互功能
- **实时屏幕捕获**：支持DXCam和Win32两种捕获方式
- **血量检测**：实时监控玩家和Boss血量变化
- **光标状态检测**：确保游戏控制的准确性
- **自动按键控制**：模拟键盘输入执行游戏动作

### 🛡️ 稳定性保障
- **异常处理**：自动处理游戏崩溃和异常情况
- **超时保护**：防止训练过程无限期卡死
- **模型备份**：自动保存训练进度和最佳模型
- **紧急重启**：多种重启策略应对不同情况

### 以上部分为CALUDE4总结

##  许可

本项目采用MIT许可证，详见LICENSE文件。

