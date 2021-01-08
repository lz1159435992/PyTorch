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
    def eval(self,env):
        return env[self.name]
    def __str__(self):
        return self.name

class Not(Expr):
    def __init__(self, name):
        self.name = name
    def eval(self,env):
        return not(self.name.eval(env))
    def __str__(self):
        return f"!{self.name}"

    
class LogOp(Expr):
    def __init__(self,left,right):
        self.left = left
        self.right = right
    def __str__(self):
        return "(" +str(self.left) + self.op +str(self.right) +")"
    def eval(self,env):
        return self.fun(self.left.eval(env),self.right.eval(env))   

class And(LogOp):

     op = '&'
     
     def fun(self,x,y):
         return x & y
    
class Or(LogOp):

     op = '|'
     
     def fun(self,x,y):
         return x | y
    
class Eq(LogOp):

     op = '=='
     
     def fun(self,x,y):
         return x == y
    
 

e1 = Or(Var("x"),Not(Var("x")))
e2 = Eq(Var("x"),Not(Not(Var("x"))))
e3 = Eq(Not(And(Var("x"),Var("y"))),Or(Not(Var("x")),Not(Var("y"))))
e4 = Eq(Not(And(Var("x"),Var("y"))),And(Not(Var("x")),Not(Var("y"))))


