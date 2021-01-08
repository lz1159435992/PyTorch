# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:58:24 2020

@author: depal
"""

class Expr:
    pass

class Var(Expr):
    def __init__(self,name):
        self.name = name
    def __str__(self):
        return self.name
    def eval(self,env):
        return env[self.name]

class Not(Expr):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"!{self.name}"
    def eval(self,env):
        return not(self.name.eval(env))
    
class LogOp(Expr):
    def __init__(self,left,right):
        self.left = left
        self.right = right
    def eval(self,env):
        return self.op(self.left.eval(env),self.right.eval(env))   

class And(LogOp):
    def __str__(self):
        return f"({self.left}&{self.right})"
    def op(self,x,y):
        return x and y
    
class Or(LogOp):
    def __str__(self):
        return f"{self.left}|{self.right}"
    def op(self,x,y):
        return x or y
    
class Eq(LogOp):
    def __str__(self):
        return f"{self.left}=={self.right}"
    def op(self,x,y):
        return x == y
    
    

env = { "x" : True, "y" : False }  

e1 = Or(Var("x"),Not(Var("x")))
e2 = Eq(Var("x"),Not(Not(Var("x"))))
e3 = Eq(Not(And(Var("x"),Var("y"))),Or(Not(Var("x")),Not(Var("y"))))
e4 = Eq(Not(And(Var("x"),Var("y"))),And(Not(Var("x")),Not(Var("y"))))
print(e1,e2,e3,e4)


