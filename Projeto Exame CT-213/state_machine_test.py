import numpy as np
from sudoku import Sudoku
from state_machine import FiniteStateMachine, Fill_Possibilities_State

sudokus_solved = 0
starting_grid = np.zeros(shape=(9, 9))
starting_grid = starting_grid.astype('int')
cont = 0
data = np.zeros(shape=(324, 3))
data = data.astype(int)
arq = open('data17(extreme).txt')
texto = arq.read()
i = 0
j = 0
last = "\n"
for elemento in texto:
    if elemento != "\t" and i != 324:
        if elemento == " ":
            if last != "number":
                data[i, j] = 0
                if j == 2:
                    j = 0
                    i += 1
                else:
                    j += 1
            last = "space"
        elif elemento == "\n":
            if last == "number":
                if j < 2:
                    data[i, (j+1):3] = 0
                i += 1
                j = 0
            elif last == "\n":
                data[i, 0:3] = 0
                i += 1
            last = "\n"
        else:
            data[i, j] = int(elemento)
            if j < 2:
                j += 1
            last = "number"
    else:
        last = "\t"

while cont < 324:
    for i in range(3):
        for j in range(3):
            starting_grid[i, 3*j: 3*j + 3] = data[cont + i + 3*j, :]
    for i in range(3):
        for j in range(3):
            starting_grid[i + 3, 3*j: 3*j + 3] = data[cont + i + 9 + 3*j, :]
    for i in range(3):
        for j in range(3):
            starting_grid[i + 6, 3*j: 3*j + 3] = data[cont + i + 18 + 3*j, :]
    print("///SUDOKU", cont//27 + 1, "INICIAL///")
    print("")
    print(starting_grid)
    behavior = FiniteStateMachine(Fill_Possibilities_State())
    sudoku_grid = Sudoku(starting_grid, behavior)
    a = True
    while a:
        a = sudoku_grid.update()
    print("")
    print("***SUDOKU", cont//27 + 1, "RESOLVIDO***")
    print("")
    print(sudoku_grid.grid)
    print("")
    flag = 1
    for i in range(9):
        for j in range(9):
            if sudoku_grid.grid[i, j] == 0:
                flag = 0
    sudokus_solved += flag
    cont += 27
print("")
print("Foram resolvidos ", sudokus_solved, "/ 12 sudokus")

# SE QUISER TESTAR UM SUDOKU:
#starting_grid = [[0, 0, 0, 0, 0, 8, 0, 5, 0],
#                 [5, 0, 0, 0, 0, 1, 4, 0, 0],
#                 [4, 9, 1, 0, 0, 0, 0, 8, 0],
#                 [7, 0, 0, 0, 6, 0, 0, 0, 8],
#                 [0, 0, 4, 0, 1, 0, 9, 0, 0],
#                 [9, 0, 0, 0, 3, 0, 0, 0, 2],
#                 [0, 4, 0, 0, 0, 0, 6, 3, 7],
#                 [0, 0, 9, 4, 0, 0, 0, 0, 5],
#                 [0, 7, 0, 5, 0, 0, 0, 0, 0]]
#behavior = FiniteStateMachine(Fill_Possibilities_State())
#sudoku_grid = Sudoku(starting_grid, behavior)
#a = True
#while a:
#    a = sudoku_grid.update()
#print(sudoku_grid.grid)