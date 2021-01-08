# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:28:58 2020

@author: BOOMBOOM
"""
Expr1 = 'x*y+7'
Expr2 = 'x*(y+7)'
class Expr:
    pass

class Var(Expr) :
    
    def __init__(self,name):
        self.name = name
    def __str__(self):
        return self.name
    def eval(self,env):
        return env[self.name]
    
class Constant(Expr) :
    
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return str(self.value)
    def eval(self,env) :
        return self.value

class Binop(Expr):
    
    def __init__(self,left,right):
        self.right = right
        self.left = left
    def __str__(self):
        return "(" +str(self.left) +\
                self.op+str(self.right) +")"
    def eval(self,env) :
        return self.fun(self.left.eval(env),\
                        self.right.eval(env))
        
class Times(Binop) :
    
    op= "*"
    
    def fun(self,x,y) :
        return x*y
    
class Plus(Binop) :
    
    op = "+"
    
    def fun(self,x,y):
        return x+y


    
expr1 = Plus(Times(Var("x"),Var("y")),
             Constant(7))

expr2 = Times(Var("x"),
              Plus(Var("y"),Constant(7)))

expr3 = Plus(Var("x"),Plus(Var("y"),Var("z")))

env = {"x" : 3, "y" : 7}
env = {"x" :3, "y" :7, "z" :5}
print(expr2)












