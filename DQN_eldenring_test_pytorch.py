# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
import directkeys
from getkeys import key_check
import random
from DQN_pytorch_gpu import DQN
import os
import pandas as pd
from restart import restart,restart2, restart3
import torch
import dxcam_cpp as dxcam
import win32gui, win32ui, win32con, win32api

USE_WIN32_FALLBACK = True

camera = None
use_dxcam = False
offset_x = 0
offset_y = 0
game_width = 1280
game_height = 720

REGIONS = {
    'self_blood': (105, 34, 632, 35),
    'boss_blood': (310, 580, 969, 580)
}

def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('开始游戏')
            time.sleep(1)
        else:
            paused = True
            print('暂停游戏')
            time.sleep(1)
    if paused:
        print('已暂停')
        while True:
            keys = key_check()
            if 'T' in keys:
                if paused:
                    paused = False
                    print('开始游戏')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused

def find_game_window_logo(frame, template_path="./logo.png", threshold=0.8):
   
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"警告：无法加载logo模板 {template_path}")
        return None
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        return max_loc
    else:
        return None

def grab_screen_win32(region=None):
    """使用win32gui的屏幕捕获方法（备用方案）"""
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)
    
    # 转换为BGR格式（去掉alpha通道）
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

def init_camera(target_fps=60):
    """初始化屏幕捕获相机"""
    global camera, use_dxcam
    
    if USE_WIN32_FALLBACK:
        print("使用win32gui屏幕捕获方法（手动设置）")
        use_dxcam = False
        camera = None
        return None
    
    try:
        camera = dxcam.create(output_color="BGR")
        camera.start(target_fps=target_fps)
        use_dxcam = True
        print("使用dxcam进行屏幕捕获")
        return camera
    except Exception as e:
        print(f"dxcam初始化失败: {e}")
        print("自动切换到win32gui屏幕捕获方法")
        use_dxcam = False
        camera = None
        return None

def grab_screen():
    """获取屏幕最新帧"""
    global camera, use_dxcam
    if use_dxcam and camera is not None:
        return camera.get_latest_frame()
    else:
        return grab_screen_win32()

def get_game_window():
    """获取游戏窗口区域"""
    global offset_x, offset_y
    frame = grab_screen()
    if frame is None:
        return None
        
    # 如果还没有定位过游戏窗口，先定位
    if offset_x == 0 and offset_y == 0:
        logo_position = find_game_window_logo(frame)
        if logo_position is None:
            print("未找到游戏窗口")
            return None
        offset_x, offset_y = logo_position
        offset_y += 43  # 调整标题栏偏移
        offset_x -= 2
        print(f"游戏窗口定位成功: ({offset_x}, {offset_y})")
    else:
        # 如果已经定位过，直接使用之前的位置
        # print("使用之前定位的游戏窗口")
        pass
    
    # 提取游戏窗口区域
    game_window = frame[offset_y:offset_y + game_height, offset_x:offset_x + game_width]
    return game_window

def process_game_image(cut_window):
    """处理游戏图像：裁切、缩放、灰度化"""
        
    # 裁切游戏画面的核心区域
    # cut_window = game_window[93:93+460, 342:342+644]
    
    # 转换为灰度图
    gray_cut_window = cv2.cvtColor(cut_window, cv2.COLOR_BGR2GRAY)
    
    # 缩放到96x88
    resized_window = cv2.resize(gray_cut_window, (96, 88))
    
    return resized_window

def process_image_cut(game_window):
    """裁切游戏画面的核心区域"""
    cut_window = game_window[93:93+460, 342:342+644]
    return cut_window

def get_self_blood(game_window):
    if game_window is None:
        return 0
        
    sx, sy, ex, ey = REGIONS['self_blood']
    gray_frame = cv2.cvtColor(game_window, cv2.COLOR_BGR2GRAY)
    row_pixels = gray_frame[sy, sx:ex]
    self_blood = 0
    for pixel_value in row_pixels:
        if 34 < pixel_value < 67:
            self_blood += 1
    return self_blood

def get_boss_blood(game_window):
    sx, sy, ex, ey = REGIONS['boss_blood']  
    red_green_channels = game_window[:, :, [2, 1]]  # 同时提取红色和绿色通道 (BGR格式)
    red_green_pixels = red_green_channels[sy, sx:ex] 
    boss_blood = 0
    for pixel in red_green_pixels:
        red_value, green_value = pixel 
        if red_value > 50 and green_value < 4:
            boss_blood += 1
    return boss_blood

def check_aim_status(cut_window):
    """检测aim状态，返回是否有aim"""
    if cut_window is None:
        return False
        
    # 裁切aim检测区域（与process_game_image相同的区域）
    # cut_window = game_window[93:93+460, 342:342+644]
    
    # 检测B通道是否有大于250的像素值
    has_aim = np.any(cut_window[:, :, 0] > 250)
    return has_aim


def handle_aim_detection(cut_window, no_aim_count):
    """处理aim检测逻辑，返回更新后的no_aim_count和光标状态"""
    has_aim = check_aim_status(cut_window)
    
    if not has_aim:
        no_aim_count += 1
        print(f"B通道没有大于250的像素 (连续第{no_aim_count}次)")
        if no_aim_count >= 3 and no_aim_count < 20:
            print(f"连续{no_aim_count}次检测到没有aim，调用lock_vision")
            directkeys.lock_vision()
            # 不再重置计数器，继续检测直到找到光标
            if no_aim_count % 17 == 0:
                print(f"尝试转头")
                directkeys.hit_O(1)
                time.sleep(0.5)
                directkeys.lock_vision()
        elif no_aim_count >= 20:
            print(f"连续{no_aim_count}次检测到没有aim，超过20次阈值")
            # 返回特殊状态，表示需要停止训练并启动restart3
    else:
        if no_aim_count > 0:
            print(f"检测到aim，重置计数器 (之前连续{no_aim_count}次未检测到)")
        no_aim_count = 0  # 只有检测到aim时才重置计数器
    
    return no_aim_count, has_aim

def wait_for_black_screen(timeout=30):
    """等待黑屏或接近黑屏的状态"""
    print("等待黑屏...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # 获取当前屏幕
            game_window = get_game_window()
            if game_window is None:
                time.sleep(0.5)
                continue
            
            # 计算图像的平均亮度
            gray_image = cv2.cvtColor(game_window, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_image)
            
            # print(f"当前平均亮度: {avg_brightness:.2f}")
            
            # 如果平均亮度低于阈值，认为是黑屏
            if avg_brightness < 9:  # 可以根据需要调整阈值
                print("检测到黑屏状态")
                return True
                
            time.sleep(0.5)
            
        except Exception as e:
            print(f"黑屏检测出错: {e}")
            time.sleep(0.5)
    
    print(f"等待黑屏超时 ({timeout}秒)")
    return False

def restart_with_black_screen_wait():
    """等待黑屏后再执行restart"""
    print("准备重启，先等待黑屏...")
    
    # 等待黑屏
    if wait_for_black_screen(timeout=30):
        print("检测到黑屏，开始重启")
        time.sleep(2)  # 额外等待2秒确保稳定
        restart()
    else:
        print("未检测到黑屏，直接重启")
        restart()

def take_action(action):
    """执行动作函数 - 4个动作空间"""
    print(f"执行动作: {action}")
  
    if action == 0:    
        # directkeys.randommove()
        # directkeys.rightdodge()
        # time.sleep(0.3)
        pass
    elif action == 1:  
        directkeys.attack()
    elif action == 2:   
        directkeys.dodge()
    elif action == 3:  
        directkeys.rightdodge()

def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break, survival_steps, action=None):
    """动作状态判断函数 - 只判断游戏状态，不计算训练奖励"""
    
    # 判断玩家是否死亡（血量低于3）
    if next_self_blood < 3:
        if emergence_break < 2:
            done = 1      # 游戏结束标志
            stop = 0      # 重置停止标志
            emergence_break += 1
            boss_defeated = False  # 玩家死亡，boss未被击败
            print(f'玩家死亡，存活{survival_steps}步')
            return done, stop, emergence_break, boss_defeated
        else:
            done = 1
            stop = 0
            emergence_break = 100
            boss_defeated = False
            return done, stop, emergence_break, boss_defeated
    
    # 判断Boss是否死亡
    elif next_boss_blood < 1:
        if emergence_break < 2:
            done = 1    # 游戏结束
            stop = 0
            emergence_break += 1
            boss_defeated = True  # 明确标识boss被击败
            print(f'Boss死亡！存活{survival_steps}步')
            return done, stop, emergence_break, boss_defeated
        else:
            done = 0
            stop = 0
            emergence_break = 100
            boss_defeated = True  # 明确标识boss被击败
            return done, stop, emergence_break, boss_defeated
    else:
        # 检测玩家受伤
        if next_self_blood - self_blood < -10:
            if stop == 0:
                stop = 1  
                print("检测到玩家受到大量伤害")
        else:
            stop = 0  
        
        # 检测Boss受伤
        if boss_blood - next_boss_blood > 10:
            print("成功对Boss造成伤害")
        
        done = 0  
        emergence_break = 0  
        boss_defeated = False  # 正常游戏状态，boss未被击败
        
        # 每100步打印一次存活信息
        if survival_steps % 100 == 0:
            print(f'存活{survival_steps}步')
            
        return done, stop, emergence_break, boss_defeated

script_dir = os.path.dirname(os.path.abspath(__file__))
DQN_model_path = os.path.join(script_dir, "model_gpu")

WIDTH = 96
HEIGHT = 88

action_size = 4

TEST_EPISODES = 100
paused = True

if __name__ == '__main__':
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，将使用CPU")
    
    # 初始化屏幕捕获
    init_camera()
    
    # 初始化DQN智能体（PyTorch版本）
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, "")
    
    # 加载预训练模型
    model_file = os.path.join(DQN_model_path, "pytorch_model.pth")
    if os.path.exists(model_file):
        if agent.load_model(model_file):
            print(f"成功加载预训练模型: {model_file}")
        else:
            print(f"加载模型失败: {model_file}")
            exit(1)
    else:
        print(f"模型文件不存在: {model_file}")
        exit(1)
    
    # 设置为推理模式（不进行探索）
    agent.epsilon = 0.0  # 关闭随机探索
    print("模型设置为推理模式（epsilon=0）")
    
    # 开始时暂停
    paused = pause_game(paused)
    
    # 紧急中断计数器
    emergence_break = 0
    restart()
    
    # 开始测试循环
    for episode in range(TEST_EPISODES):
        print(f"\n=== 开始测试回合 {episode + 1}/{TEST_EPISODES} ===")
        
        # 获取游戏窗口
        game_window = get_game_window()
        if game_window is None:
            print("无法获取游戏窗口，跳过此回合")
            continue
        
        cut_window = process_image_cut(game_window)
        # 处理游戏图像
        station = process_game_image(cut_window)
        if station is None:
            print("无法处理游戏图像，跳过此回合")
            continue
            
        # 获取初始血量
        boss_blood = get_boss_blood(game_window)
        self_blood = get_self_blood(game_window)
        
        print(f"回合 {episode + 1} 开始 - Boss血量: {boss_blood}, 玩家血量: {self_blood}")
        
        done = 0
        stop = 0
        no_aim_count = 0
        survival_steps = 0
        last_time = time.time()
        boss_defeated = False

        episode_start_time = time.time()
        timeout_triggered = False
        
        # 单回合测试循环
        while True:
            # 检查是否超时（5分钟 = 300秒）
            current_time = time.time()
            if current_time - episode_start_time > 300: 
                print(f"回合 {episode + 1} 超时，触发重启")
                timeout_triggered = True
                done = 1
                break
            
            station = process_game_image(process_image_cut(get_game_window()))
            station_tensor = torch.from_numpy(station).float().unsqueeze(0).unsqueeze(0)
            
            print('循环耗时 {} 秒'.format(time.time()-last_time))
            last_time = time.time()
              
            action = agent.choose_action(station_tensor)
            take_action(action)
            if action != 0:
                survival_steps += 1
            
            game_window = get_game_window()
            next_cut_window = process_image_cut(game_window)
            next_station = process_game_image(next_cut_window)
            if next_station is None:
                break

            next_boss_blood = get_boss_blood(game_window)
            next_self_blood = get_self_blood(game_window)
            
            no_aim_count, has_aim = handle_aim_detection(next_cut_window, no_aim_count)
            
            if no_aim_count >= 30:
                print(f"光标失去超过30次({no_aim_count}次),停止测试并启动restart3")
                restart3()
                done = 0
                break
            
            print(f"血量变化 - Boss: {boss_blood}->{next_boss_blood}, 玩家: {self_blood}->{next_self_blood}")
            
            done, stop, emergence_break, boss_defeated = action_judge(boss_blood, next_boss_blood,
                                                                     self_blood, next_self_blood,
                                                                     stop, emergence_break, survival_steps, action)

            print(f"动作: {action}")

            if boss_defeated:
                print("检测到boss死亡！")

            if emergence_break == 100:
                print("紧急中断")
                paused = True
            
            # 更新状态和血量信息
            self_blood = next_self_blood
            boss_blood = next_boss_blood
            
            paused = pause_game(paused)
            if done == 1:
                print("回合结束")
                break
        
        # 处理超时情况
        if timeout_triggered:
            print("执行超时处理流程：使用restart3进行紧急重启")
            restart3()
            continue

        episode_time = time.time() - episode_start_time
        print(f'回合: {episode + 1}, 存活步数: {survival_steps}, 回合时长: {episode_time:.2f}秒')
        
        if boss_defeated:
            print("Boss被击败，使用restart2进行重启")
            restart2()
        elif done == 1:
            print("玩家死亡或其他原因，使用带黑屏等待的重启")
            restart_with_black_screen_wait()
        time.sleep(1)
    
    print(f"\n=== 测试完成，共进行了 {TEST_EPISODES} 回合 ===")