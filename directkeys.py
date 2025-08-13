# -*- coding: utf-8 -*-

import ctypes
import time
import random

SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13
V = 0x2F

Q = 0x10
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21
G = 0x22
E = 0x12

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01
enter = 0x1C

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def hit_E():
    PressKey(E)
    time.sleep(0.05)
    ReleaseKey(E)
    time.sleep(0.1)

def hit_enter():
    PressKey(enter)
    time.sleep(0.05)
    ReleaseKey(enter)
    time.sleep(0.1)

def hit_G():
    PressKey(G)
    time.sleep(0.05)
    ReleaseKey(G)
    time.sleep(0.1)

def hit_O(t):
    PressKey(O)
    time.sleep(t)
    ReleaseKey(O)
    time.sleep(0.1)

def hit_F():
    PressKey(F)
    time.sleep(0.05)
    ReleaseKey(F)
    time.sleep(0.1)

def defense():
    PressKey(M)
    time.sleep(0.7)
    ReleaseKey(M)
    time.sleep(0.4)
    
def attack():
    PressKey(J)
    time.sleep(0.03)
    ReleaseKey(J)
    # time.sleep(0.7)
    
def go_forward(t):
    PressKey(W)
    time.sleep(t)
    ReleaseKey(W)
    
def go_back(t):
    PressKey(S)
    time.sleep(t)
    ReleaseKey(S)
    
def go_left(t):
    PressKey(A)
    time.sleep(t)
    ReleaseKey(A)
    
def go_right(t):
    PressKey(D)
    time.sleep(t)
    ReleaseKey(D)
    
def jump():
    PressKey(K)
    time.sleep(0.1)
    ReleaseKey(K)
    #time.sleep(0.1)
    
def dodge():
    PressKey(W)
    time.sleep(0.02)
    PressKey(K)
    time.sleep(0.02)
    ReleaseKey(K)
    time.sleep(0.02)
    ReleaseKey(W)
    # time.sleep(0.2)

def rightdodge():
    PressKey(D)
    time.sleep(0.02)
    PressKey(K)
    time.sleep(0.02)
    ReleaseKey(K)
    time.sleep(0.02)
    ReleaseKey(D)
    # time.sleep(0.2)

    
def lock_vision():
    PressKey(V)
    time.sleep(0.3)
    ReleaseKey(V)
    time.sleep(0.1)
    
def go_forward_QL(t):
    PressKey(W)
    time.sleep(t)
    ReleaseKey(W)
    
def turn_left(t):
    PressKey(left)
    time.sleep(t)
    ReleaseKey(left)
    time.sleep(0.1)
    
def turn_up(t):
    PressKey(up)
    time.sleep(t)
    ReleaseKey(up)
    time.sleep(0.1)
    
def turn_right(t):
    PressKey(right)
    time.sleep(t)
    ReleaseKey(right)
    time.sleep(0.1)

def turn_down():
    PressKey(down)
    time.sleep(0.05)
    ReleaseKey(down)
    time.sleep(0.1)
    
def F_go():
    PressKey(F)
    time.sleep(0.5)
    ReleaseKey(F)
    
def forward_jump(t):
    PressKey(W)
    time.sleep(t)
    PressKey(K)
    ReleaseKey(W)
    ReleaseKey(K)
    
def press_esc():
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)
    
def dead():
    PressKey(M)
    time.sleep(0.5)
    ReleaseKey(M)

def randommove():
    """随机左移或右移0.5秒"""
    # 随机选择左移(0)或右移(1)
    direction = random.choice([0, 1])
    
    if direction == 0:
        # 左移0.5秒
        PressKey(A)
        time.sleep(0.5)
        ReleaseKey(A)
        print("随机移动：左移0.5秒")
    else:
        # 右移0.5秒
        PressKey(D)
        time.sleep(0.5)
        ReleaseKey(D)
        print("随机移动：右移0.5秒")

if __name__ == '__main__':
    time.sleep(5)
    
    while(True):
        attack()
        time.sleep(1)


        
    
    # PressKey(W)
    # time.sleep(0.4)
    # ReleaseKey(W)
    # time.sleep(1)
    
    # PressKey(J)
    # time.sleep(0.1)
    # ReleaseKey(J)
    # time.sleep(1)