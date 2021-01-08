# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:53:40 2020

@author: Jingyi Yue
"""


class Expr:  # abstract class
    # Create dictionary and list to store letters and their truth values
    Dict = {}
    Lst = []
    A = ''
    Tauto = []

    # Use the set() to create an unordered set of unique elements
    def __set__(self):
        a = self.__str__()
        '''
        replaced all the operations with an empty space, later I can use len() to calculate the length 
        and iterate through the output
        '''
        a = a.replace('(', ' ')
        a = a.replace(')', ' ')
        a = a.replace('!', ' ')
        a = a.replace('&', ' ')
        a = a.replace('|', ' ')
        a = a.replace('==', ' ')
        b = a.split(' ')  # use split() to separate all the operators
        set1 = set(b)
        # 除去多余的''
        if '' in set1:
            set1.remove('')
        self.Lst = list(set1)  # self.Lst will only get x,y or p,q,r variables

    def __Count__(self, i):  # judge whether it will print results or enter recursion
        # If the end has been reached, print and calculate
        if i == len(self.Lst):
            self.A = self.A + str(self.eval(self.Dict))
            B = ''
            for j in self.Lst:
                if str(self.Dict[j]) == 'True':  # Make the program more concise and clear
                    B = B + str(self.Dict[j]) + '|'
                elif str(self.Dict[j]) == 'False':
                    B = B + str(self.Dict[j]) + '|'
            print(B, self.eval(self.Dict))
            self.Tauto.append(self.eval(self.Dict))
            return
        '''
        If the end hasn't been reached, use recursion for assignmentassign, 
        assign True and False to each variable
        '''
        self.Dict[self.Lst[i]] = True
        self.__Count__(i + 1)  # through this line to go thtough the next position
        self.Dict[self.Lst[i]] = False
        self.__Count__(i + 1)

    def make_tt(self):  # through this method to print the first line o the whole truth table(e.g. x,y,p,q,r)

        self.__set__()
        first_line = ''
        for i in range(0, len(self.Lst)):
            c = self.Lst[i]
            first_line += c + '   | '

        print(first_line, self.__str__())
        self.__Count__(0)
    def isTauto(self):
        flag = True
        for i in self.Tauto:
            flag = flag and i
        return flag
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

    def sett(self):
        return self.name.sett()

    def sets(self):
        return list(set(str(self.sett())))
class LogOp(Expr):
    level = -1
    def __init__(self,left,right):
        self.left = left
        self.right = right
        # self.s = self.sets()
        # self.env = {}
        # self.truth = []
        # self.makettlst = []
        # self.tauto = []
        
    def __str__(self):
        if self.left.level < self.level and self.right.level < self.level:
            return "(" + str(self.left) +")"+ self.symbol + "(" +str(self.right)+ ")"
        elif self.left.level >= self.level and self.right.level < self.level:
            return str(self.left) + self.symbol + "(" + str(self.right) + ")"
        elif self.left.level < self.level and self.right.level >= self.level:
            return "(" + str(self.left) + ")" + self.symbol + str(self.right)
        elif self.left.level >= self.level and self.right.level >= self.level:
            return str(self.left) + self.symbol + str(self.right)
        
    # def eval(self,env):
    #     return self.fun(self.left.eval(env),self.right.eval(env))
    #
    # def sort(self,n,tab=[]):
    #     if not n:
    #         for i in range(len(self.s)):
    #             self.env[self.s[i]] = tab[i]
    #         self.truth = [str(tab[i]) for i in range(len(tab))]
    #         self.truth +=[str(self.eval(self.env))]
    #         self.makettlst +=[self.truth]
    #         return self.makettlst
    #     else:
    #         for i in [True,False]:
    #             self.sort(n-1,tab+[i])
    # def make_tt(self):
    #     n = len(self.s)
    #     string = ''
    #     print("\t\t|".join(self.s)+"\t\t|"+self.__str__())
    #     self.sort(n)
    #     for i in self.makettlst:
    #         for j in range(len(i)):
    #             if j != len(i)-1:
    #                 string += f"{i[j]}"+'\t' +'|'
    #             else:
    #                 string +=''+str(i[j]) + "\n"
    #     return string
    # def sett(self):
    #     return self.left.sett()+self.right.sett()
    #
    # def sets(self):
    #     return list(set(str(self.sett())))
    # def lst_tauto(self,n,tab=[]):
    #     if not n:
    #         for i in range(len(self.s)):
    #             self.env[self.s[i]]=tab[i]
    #         self.tauto+=[self.eval(self.env)]
    #     else:
    #         for i in [True,False]:
    #             self.lst_tauto(n-1,tab+[i])
    # def isTauto(self):
    #     n = len(self.s)
    #     self.lst_tauto(n)
    #     for i in range(len(self.tauto)):
    #         if self.tauto[i]==self.tauto[i+1]:
    #             return True
    #         return False
    

class And(LogOp):

    level = 3
    symbol = "&"

    def eval(self, env):
        return self.left.eval(env) & self.right.eval(env)
class Or(LogOp):

    level = 2
    symbol = "|"
    def eval(self, env):
        return self.left.eval(env) | self.right.eval(env)
class Eq(LogOp):

    level = 1
    symbol = "=="
    def eval(self,env) :
        return self.left.eval(env) == self.right.eval(env)

e1 = Or(Var("x"),Not(Var("x")))
e2 = Eq(Var("x"),Not(Not(Var("x"))))
e3 = Eq(Not(And(Var("x"),Var("y"))),Or(Not(Var("x")),Not(Var("y"))))
e4 = Eq(Not(And(Var("x"),Var("y"))),And(Not(Var("x")),Not(Var("y"))))

e5 = And(Not(Var("p")),Var("q"))
e6 = Not(And(Var("p"),Var("q")))
e7 = Or(And(Var("p"),Var("q")),Var("r"))
e8 = And(Var("p"),Or(Var("q"),Var("r")))
e9 = Eq(Or(Var("p"),Var("q")),Var("r"))
e10 = Or(Var("p"),Eq(Var("q"),Var("r")))
e11 = And(Var("x"),And(Var("y"),Var("z")))
e12 = And(And(Var("x"),Var("y")),Var("z"))


print(e1)
print(e2)
print(e3)
print(e4)
print(e5)
print(e6)
print(e7)
print(e8)
print(e9)
print(e10)
print(e11)
print(e12)
print(e7.make_tt())
# print(e2.make_tt())
# print(e1.isTauto())
print(e7.isTauto())
#print(e4.make_tt)
