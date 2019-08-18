import numpy as np

class Sudoku(object):
    """
    Represents a sudoku grid.
    """
    def __init__(self, starting_grid, behavior):
        """
        Creates an unfinished sudoku grid.

        :param starting_grid: the given sudoku to evaluate.
        :type starting_grid: bidimensional numpy array.
        """
        self.grid = starting_grid
        self.behavior = behavior
        self.dimension = len(starting_grid[0])
        self.type = int(np.sqrt(self.dimension))
        self.possibilities = np.zeros((self.dimension, self.dimension, self.dimension))
        self.possibilities_line = np.ones((self.type, self.type, self.dimension, self.type))
        self.possibilities_column = np.ones((self.type, self.type, self.dimension, self.type))

    def square_of_the_cell(self, cell):
        """
        Obtains square in which a certain cell is contained.

        :param cell: cell in which will be verified.
        :type cell: int tuple.
        :return: the position of the bigger square.
        :rtype: int tuple.
        """
        line = cell[0]
        column = cell[1]
        position = np.zeros(2, int)
        position[0] = line//self.type
        position[1] = column//self.type
        return position

    def is_number_valid(self, number, cell):
        """
        Verifies if a number can be in an certain cell given the current knowledge.

        :param number: number to be verified.
        :type number: int.
        :param cell: cell in which will be verified.
        :type cell: int tuple.
        :return: the validity of the number in the cell.
        :rtype: bool.
        """
        if self.grid[cell[0]][cell[1]] != 0:
            return False
        line = cell[0]
        column = cell[1]
        square = self.square_of_the_cell(cell)
        is_it = True
        for cell_in_line in range(self.dimension):
            if cell_in_line != column and self.grid[line][cell_in_line] == number:
                is_it = False
        for cell_in_column in range(self.dimension):
            if cell_in_column != line and self.grid[cell_in_column][column] == number:
                is_it = False
        for cell_i in range(self.type):
            for cell_j in range(self.type):
                position = (cell_i + square[0]*self.type, cell_j + square[1]*self.type)
                if self.grid[position[0]][position[1]] == number and position != (line, column):
                    is_it = False
        return is_it

    def number_of_possible_numbers(self, cell):
        """
        Obtains the number of the possible numbers in a certain cell given the current knowledge

        :param cell: cell in which will be verified.
        :type cell: int tuple.
        :return: number of the possible numbers.
        :rtype: int.
        """
        count = 0
        possible_numbers = self.possible_numbers(cell)
        for number in range(1, self.dimension + 1):
            if possible_numbers[number - 1]:
                count += 1
        return count

    def possible_numbers(self, cell):
        """
        Obtains all the possible numbers in a certain cell given the current knowledge.

        :param cell: cell in which will be verified.
        :type cell: int tuple.
        :return: boolean array with the possible numbers.
        :rtype: bool array.
        """
        array = np.zeros(self.dimension)
        if self.grid[cell[0]][cell[1]] != 0:
            array[self.grid[cell[0]][cell[1]] - 1] = 1
        else:
            for number in range(1, self.dimension + 1):
                if self.is_number_valid(number, cell):
                    array[number-1] = 1
        return array

    def update_possibilities(self, number, cell):
        """
        Updates all new the possibilities from the grid that might change due to an update in the grid.

        :param number: number recently put in the grid.
        :type number: int.
        :param cell: cell in which will be verified.
        :type cell: int tuple.
        """
        for i in range(self.dimension):
            if i != cell[0]:
                self.possibilities[i, cell[1], number - 1] = 0
        for j in range(self.dimension):
            if j != cell[1]:
                self.possibilities[cell[0], j, number - 1] = 0
        square = self.square_of_the_cell(cell)
        for i in range(self.type):
            for j in range(self.type):
                if i + square[0]*self.type == cell[0] and j + square[1]*self.type == cell[1]:
                    self.possibilities[cell[0], cell[1], :] = 0
                    self.possibilities[cell[0], cell[1], number - 1] = 1
                else:
                    self.possibilities[i + square[0]*self.type, j + square[1]*self.type, number - 1] = 0

    def line_possibility(self, square):
        """
        Updates the possibilities_line array. It represents the lines in a square in which is possible to any number be.

        :param square: square to be analysed.
        :type square: int tuple.
        """
        for number in range(1, self.dimension + 1):
            for line_inside in range(self.type):
                is_number_possible_in_line = 0
                for column_inside in range(self.type):
                    if self.possibilities[square[0]*self.type + line_inside, square[1]*self.type + column_inside, number - 1] == 1:
                        is_number_possible_in_line = 1
                self.possibilities_line[square[0], square[1], number - 1, line_inside] = is_number_possible_in_line

    def column_possibility(self, square):
        """
        Updates the possibilities_column array. It represents the columns in a square in which is possible to any number be.

        :param square: square to be analysed.
        :type square: int bidimensional array.
        """
        for number in range(1, self.dimension + 1):
            for column_inside in range(self.type):
                is_number_possible_in_column = 0
                for line_inside in range(self.type):
                    if self.possibilities[square[0]*self.type + line_inside, square[1]*self.type + column_inside, number - 1] == 1:
                        is_number_possible_in_column = 1
                self.possibilities_line[square[0], square[1], number - 1, column_inside] = is_number_possible_in_column

    def update(self):
        """
        Updates the sudoku, including its behavior.
        """
        a = self.behavior.update(self)
        return a


