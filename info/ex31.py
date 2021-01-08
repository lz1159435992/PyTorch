# to check whether position (x,y) will not be attackted by other queens.
def is_attack(queen, x, y):
    for i in range(x):
        if queen[i] == y or abs(x - i) == abs(queen[i] - y):     #to check whether the queen is in the same column or diagonal
            return True
    return False       #the queen will not be attacked
def print_grid(n,queen):
    for j in range(n):
        for k in range(n):
            if queen[k] == j:
                print(' Q ', end=' ')
            else:
                print(' . ', end=' ')
        print('\n')  # to change the line.
    input('more?')
# to put queens in position according to the column.
def solve(n, queen, col):
    for i in range(n):
        if not is_attack(queen, col, i):
            queen[col] = i
            if col == n - 1:     # The last queen is posited correctly,print the result now.
                # print(queen)
                print_grid(n,queen)
            else:
                solve(n, queen, col + 1)

def main(n):
    queen = []
    for i in range(n):
        queen.append(None)    #to create a '1 x n' list.
    solve(n, queen, 0)

main(8)