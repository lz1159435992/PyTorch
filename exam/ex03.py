# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:53:37 2020

@author: BOOMBOOM
"""
def make_empty_grid():
    global grid
    grid=[]
    for i in range(Size):
        row=[]
        for j in range(Size):
            row = row + [0]
        grid= grid + [row]
    
def print_grid():
    global grid
    for row in range(Size) :
        for col in range(Size) :
            if grid[row][col] == 0 :
                print(".",end=" ")
            else :
                print("Q",end=" ")
        print()
        

def possible(y,x,n):
    for i in range(0,Size) :
        if grid[y][i]==1 :
            return False
    for i in range(0,Size) :
        if grid[i][x]==1 :
            return False
#缺少对角线的判断        



# def solve():
#     for i in range(0,9) :
#         if grid[i] == 0 :
#             for n in range (1,10)
 #循环部分。   
    
    
 

grid=[]
create()
solve(0)