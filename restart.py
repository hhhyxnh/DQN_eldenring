import time
from directkeys import (
    go_forward, go_left, go_right, go_back,       # 你已有的移动
    turn_up, turn_left, hit_enter,hit_E,turn_down,hit_G,hit_F,                  # 镜头
    lock_vision,                           # 锁视角
    PressKey, ReleaseKey                   # 原始按键
)

def restart():
    print('将在10秒后重启')
    for i in range(10):
        t = 10 -i
        time.sleep(1)
    print("重启开始")

    go_forward(0.5)
    go_left(0.3)
    hit_E()
    time.sleep(5)
    turn_up(0.1)
    
    turn_up(0.1)
    
    hit_enter()
    time.sleep(1)
    hit_enter()
    time.sleep(1)

    turn_down()

    hit_enter()
    time.sleep(1)

    for _ in range(7):
        turn_up(0.05)
    hit_enter()
    time.sleep(1)

    # 第 5 行 ── ↓  ↩
    turn_down()
    hit_enter()
    time.sleep(1)

    # 第 6 行 ── ←  ↩
    turn_left(0.05)
    hit_enter()
    time.sleep(10)

    # 第 7 行 ── W×6s
    go_forward(2.5)
    time.sleep(3)

    # 第 8 行 ── S×6s
    go_back(4.5)
    lock_vision()

    print("重启完成")

def restart2():
 
    print('将在8秒后重启')
    for i in range(8):
        t = 8 -i
        
        time.sleep(1)
    print("重启开始")

    hit_G()
    time.sleep(2)
    hit_F()
    time.sleep(1)
    hit_E()
    time.sleep(1)
    hit_enter()
    time.sleep(8)

    

    

    # 第 1 行 ── Wx2  Ax1  ↩  E
    go_forward(0.5)          #
    go_left(0.3)             #            # 回车
    hit_E()
    time.sleep(5)
    # 第 2 行 ── ↑ ↑  ↩  D
    turn_up(0.1)
    
    turn_up(0.1)
    
    hit_enter()
    time.sleep(1)
    hit_enter()
    time.sleep(1)             # D 默认 0.4 秒

    # 第 3 行 ── ↓  ↩
    turn_down()

    hit_enter()
    time.sleep(1)

    # 第 4 行 ── ↑×7  ↩
    for _ in range(7):
        turn_up(0.05)
    hit_enter()
    time.sleep(1)

    turn_down()
    turn_down()
    hit_enter()
    time.sleep(1)

    turn_left(0.05)
    hit_enter()
    time.sleep(2)

    # 第 5 行 ── ↓  ↩
    turn_down()
    hit_enter()
    time.sleep(1)

    # 第 6 行 ── ←  ↩
    turn_left(0.05)
    hit_enter()
    time.sleep(10)

    # 第 7 行 ── W×6s
    go_forward(2.5)
    time.sleep(3)

    # 第 8 行 ── S×6s
    go_back(4.5)
    lock_vision()

    print("重启完成")

def restart3():
    print('将在8秒后重启')
    for i in range(8):
        t = 8 -i
        
        time.sleep(1)
    print("重启开始")

    hit_G()
    time.sleep(2)
    hit_F()
    time.sleep(1)
    hit_E()
    time.sleep(1)
    hit_enter()
    time.sleep(8)

    

    

    # 第 1 行 ── Wx2  Ax1  ↩  E
    go_forward(0.5)          #
    go_left(0.3)             #            # 回车
    hit_E()
    time.sleep(5)
    # 第 2 行 ── ↑ ↑  ↩  D
    turn_up(0.1)
    
    turn_up(0.1)
    
    hit_enter()
    time.sleep(1)
    hit_enter()
    time.sleep(1)             # D 默认 0.4 秒

    # 第 3 行 ── ↓  ↩
    turn_down()

    hit_enter()
    time.sleep(1)

    # 第 4 行 ── ↑×7  ↩
    for _ in range(7):
        turn_up(0.05)
    hit_enter()
    time.sleep(1)


    # 第 5 行 ── ↓  ↩
    turn_down()
    hit_enter()
    time.sleep(1)

    # 第 6 行 ── ←  ↩
    turn_left(0.05)
    hit_enter()
    time.sleep(10)

    # 第 7 行 ── W×6s
    go_forward(2.5)
    time.sleep(2)

    # 第 8 行 ── S×6s
    go_back(4.5)
    lock_vision()

    print("重启完成")

if __name__ == "__main__":  
    time.sleep(2)
    restart2()