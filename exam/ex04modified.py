# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 02:51:47 2020

@author: cqq
"""

class Expr:    # abstract class
    # Create dictionary and list to store letters and their truth values
    Dict = {}    
    Lst = []
    A = ''
    
    # Use the set() to create an unordered set of unique elements
    def __set__(self):
        a = self.__str__()
        '''
        replaced all the operations with an empty space, later I can use len() to calculate the length 
        and iterate through the output
        '''
        a = a.replace('(', ' ')
        a = a.replace(')',' ')
        a = a.replace('!',' ')
        a = a.replace('&',' ')
        a = a.replace('|',' ')
        a = a.replace('==',' ')
        b = a.split(' ')      # use split() to separate all the operators
        set1 = set(b)
        self.Lst=list(set1)   # self.Lst will only get x,y or p,q,r variables
        
    def __Count__(self,i): # judge whether it will print results or enter recursion
    # If the end has been reached, print and calculate
        if i == len(self.Lst):  
            self.A = self.A +str(self.eval(self.Dict))
            B = ''
            for j in self.Lst:
                if str(self.Dict[j])=='True':  # Make the program more concise and clear
                    B = B + str(self.Dict[j]) + '|'   
                elif str(self.Dict[j])=='False':
                    B = B + str(self.Dict[j]) + '|'   
            print(B,self.eval(self.Dict))
            return 
        '''
        If the end hasn't been reached, use recursion for assignmentassign, 
        assign True and False to each variable
        '''
        self.Dict[self.Lst[i]] = True    
        self.__Count__(i+1)  # through this line to go thtough the next position
        self.Dict[self.Lst[i]] = False
        self.__Count__(i+1)
        
    def make_tt(self):  # through this method to print the first line o the whole truth table(e.g. x,y,p,q,r)
    
        self.__set__()
        first_line = ''
        for i in range(0,len(self.Lst)):
            c = self.Lst[i]
            first_line += c + '   | '
            
        print(first_line,self.__str__())
        self.__Count__(0)
        
      
class Var(Expr) :       
    def __init__(self,name) :
        self.name = name
        
        
    def eval(self,env) :
        return env[self.name]
    
    
    def __str__(self) :
        return self.name
    
        
    
class BinOp(Expr) : # abstract class
    def __init__(self,left,right) :
        self.left = left
        self.right = right

class Not(Expr):  
    '''
    not is different from other operations like And and Or etc,
    it cannot directly return left + '!=' + right 
    '''
    def __init__(self,name):
        self.name = name
    
    def eval(self,env):
        return not(self.name.eval(env))  
    
    def __str__(self) :  # use if-else statement to divide the priority between these symbols
        if type(self.name) == And:
            return f"!({self.name})"
        elif type(self.name) == Or:
            return f"!({self.name})"
        elif type(self.name) == Eq:
            return f"!({self.name})"
        else:
            return "!{}".format(self.name)  
   
    
class And(BinOp) :  # to achieve & operation
    def eval(self,env) :
        return self.left.eval(env) & self.right.eval(env)
    
    def __str__(self) :
        Left = f"{self.left}"
        Right = f"{self.right}"
        if type(self.left)== Or or type(self.left)== Eq:
            Left = f"({self.left})"
        elif type(self.right)== Or or type(self.left)== Eq:
            Right = f"({self.right})"
        return Left + '&' + Right 
    
    
class Or(BinOp) :  # to achieve | operation
    def eval(self,env) :
        return self.left.eval(env) | self.right.eval(env)
    
    
    def __str__(self) :
        Left = f"{self.left}"
        Right = f"{self.right}"
        if type(self.left)== Eq:
            Left = f"({self.left})"
        elif type(self.right)== Eq:
            Right = f"({self.right})"
        return Left + '|' + Right 
    
class Eq(BinOp) :    # to achieve == operation
    def eval(self,env) :
        return self.left.eval(env) == self.right.eval(env)
    
    
    def __str__(self) :
        return f"{self.left}=={self.right}"

e1 = Or(Var("x"),Not(Var("x")))
e2 = Eq(Var("x"),Not(Not(Var("x"))))
e3 = Eq(Not(And(Var("x"),Var("y"))),Or(Not(Var("x")),Not(Var("y"))))
e4 = Eq(Not(And(Var("x"),Var("y"))),And(Not(Var("x")),Not(Var("y"))))
e5 = Eq(Eq(Eq(Var("p"),Var("q")),Var("r")),Eq(Var("p"),Eq(Var("q"),Var("r"))))

print(e1)
print(e2)
print(e3)
print(e4)
print(e5)

print(And(Not(Var("p")),Var("q")))
print(Not(And(Var("p"),Var("q"))))
print(Or(And(Var("p"),Var("q")),Var("r")))
print(And(Var("p"),Or(Var("q"),Var("r"))))
print(Eq(Or(Var("p"),Var("q")),Var("r")))
print(Or(Var("p"),Eq(Var("q"),Var("r"))))

print (e2.eval({"x" : True}))
print (e3.eval({"x" : True, "y" : True}))
print (e4.eval({"x" : False, "y" : True}))

print(e1.make_tt())
print(e2.make_tt())
print(e3.make_tt())
print(e4.make_tt())
print(e5.make_tt())

print (And(Var("x"),And(Var("y"),Var("z"))))
print (And(And(Var("x"),Var("y")),Var("z")))

# print (e1.isTauto())
# print (e2.isTauto())
# print (e3.isTauto())
# print (e4.isTauto())
# print (e5.isTauto())