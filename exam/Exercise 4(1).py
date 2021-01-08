# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:58:24 2020

@author: depal
"""

class Expr:
    pass

class Var(Expr):
    level = 5
    def __init__(self,name):
        self.name = name
    def eval(self,env):
        return env[self.name]
    def __str__(self):
        return self.name

class Not(Expr):
    level = 5
    def __init__(self, name):
        self.name = name
    def eval(self,env):
        return not(self.name.eval(env))
    def __str__(self):
        if self.name.level< self.level:
            return f"!({self.name})"
        else:
            return f"!{self.name}"

    
class LogOp(Expr):
    level = -1
    def __init__(self,left,right):
        self.left = left
        self.right = right
    def __str__(self):
        # if self.left.level == 0:
        #     if self.right.level ==0:
        #         return str(self.left) + self.symbol + str(self.right)
        # elif self.right.level ==0:
        #     return
        if self.left.level< self.level and self.right.level < self.level:
            return "("+str(self.left) +")"+ self.symbol + "("+str(self.right)+")"
        elif self.left.level >= self.level and self.right.level < self.level:
            return str(self.left) + self.symbol + "("+str(self.right)+")"
        elif self.left.level < self.level and self.right.level >= self.level:
            return "("+str(self.left)+")" + self.symbol + str(self.right)
        elif self.left.level >= self.level and self.right.level >= self.level:
            return str(self.left) + self.symbol + str(self.right)
    def eval(self,env):
        return self.fun(self.left.eval(env),self.right.eval(env))   
    
#    def str_aux(self,level):
#        s = self.left.str_aux(self.braket) + self.symbol + self.right.str_aux(self.braket)
#        if self.prec < level:
#            return "(" + s + ")"
#        else:
#            return s

class And(LogOp):
    
    level = 3
    symbol = "&"
#     def fun(self,x,y):
#         return x & y
    
class Or(LogOp):
    
     level = 2
     symbol = "|"

#     op = '|'
     
#     def fun(self,x,y):
#         return x | y
    
class Eq(LogOp):

     level = 1
     symbol = "=="
#     op = '=='
     
#    def fun(self,x,y):
#         return x == y
e1 = And(Or(Var("q"),Var("r")),Or(Var("q"),Var("r")))
e2 = Eq(Var("x"),Not(Not(Var("x"))))
e3 = Eq(Not(And(Var("x"),Var("y"))),Or(Not(Var("x")),Not(Var("y"))))

e5 = And(Not(Var("p")),Var("q"))
e6 = Not(And(Var("p"),Var("q")))
print(e1,e2,e3,e5,e6)
print(And(Or(Var("q"),Var("r")),Or(Var("q"),Var("r"))))
print(Or(Var("p"),Eq(Var("q"),Var("r"))))