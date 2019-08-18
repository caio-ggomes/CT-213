import numpy as np


def compare(npvector3, i, j, array):
    """
    Function that compares a tridimensional vector using his first two indices (i, j) with an array.

    :param npvector3: Tridimensional vector to be compared.
    :type npvector3: Tridimensional numpy array.
    :param i: First indice of the tridimensional vector.
    :type i: int
    :param j: Second indice of the tridimensional vector.
    :type j: int
    :param array: Array to be compared with
    :type array: numpy array
    :return: whether they are equal or not
    :rtype: bool
    """
    for k in range(len(array)):
        if npvector3[i, j, k] != array[k]:
            return False
    return True


class FiniteStateMachine(object):
    """
    A finite state machine.
    """

    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)
        if self.state.state_name == "End":
            return False
        return True


class State(object):
    """
    Abstract state class.
    """

    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


class Fill_Possibilities_State(State):
    def __init__(self):
        super().__init__("Fill_Possibilities")
        self.position = (0, 0)

    def check_transition(self, agent, state_machine):
        if self.position == (-1, -1):
            state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        cell = np.copy(self.position)
        agent.possibilities[cell[0], cell[1], :] = agent.possible_numbers(cell)[:]
        if cell[1] == agent.dimension - 1:
            if cell[0] == agent.dimension - 1:
                self.position = (-1, -1)
            else:
                self.position = (cell[0] + 1, 0)
        else:
            self.position = (cell[0], cell[1] + 1)
        pass


class Fill_Numbers_State(State):
    def __init__(self):
        super().__init__("Fill_Numbers")
        self.flag = 0
        self.position = (0, 0)

    def check_transition(self, agent, state_machine):
        if self.position == (-1, -1):
            if self.flag == 0:
                state_machine.change_state(Fill_Line_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        cell = np.copy(self.position)
        if agent.number_of_possible_numbers(cell) == 1:
            for number in range(1, agent.dimension + 1):
                if agent.possibilities[cell[0], cell[1], number - 1] == 1 and agent.grid[cell[0]][cell[1]] == 0:
                    agent.grid[cell[0]][cell[1]] = number
                    agent.update_possibilities(number, cell)
                    self.flag = 1
        if cell[1] == agent.dimension - 1:
            if cell[0] == agent.dimension - 1:
                self.position = (-1, -1)
            else:
                self.position = (cell[0] + 1, 0)
        else:
            self.position = (cell[0], cell[1] + 1)
        pass


class Fill_Line_State(State):
    def __init__(self):
        super().__init__("Fill_Line")
        self.number = 1
        self.line = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.number == -1:
            if self.flag == 0:
                state_machine.change_state(Fill_Column_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        count = 0
        for j in range(agent.dimension):
            if agent.possibilities[self.line, j, self.number - 1] == 1:
                count += 1
        if count == 1:
            for j in range(agent.dimension):
                if agent.possibilities[self.line, j, self.number - 1] == 1 and agent.grid[self.line][j] == 0:
                    agent.grid[self.line][j] = self.number
                    agent.update_possibilities(self.number, (self.line, j))
                    self.flag = 1
        if self.line == agent.dimension - 1:
            if self.number == agent.dimension:
                self.number = -1
            else:
                self.line = 0
                self.number += 1
        else:
            self.line += 1
        pass


class Fill_Column_State(State):
    def __init__(self):
        super().__init__("Fill_Column")
        self.number = 1
        self.column = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.number == -1:
            if self.flag == 0:
                state_machine.change_state(Fill_Square_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        count = 0
        for i in range(agent.dimension):
            if agent.possibilities[i, self.column, self.number - 1] == 1:
                count += 1
        if count == 1:
            for i in range(agent.dimension):
                if agent.possibilities[i, self.column, self.number - 1] == 1 and agent.grid[i][self.column] == 0:
                    agent.grid[i][self.column] = self.number
                    agent.update_possibilities(self.number, (i, self.column))
                    self.flag = 1
        if self.column == agent.dimension - 1:
            if self.number == agent.dimension:
                self.number = -1
            else:
                self.column = 0
                self.number += 1
        else:
            self.column += 1
        pass


class Fill_Square_State(State):
    def __init__(self):
        super().__init__("Fill_Square")
        self.number = 1
        self.square = [0, 0]
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.number == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Line_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        count = 0
        for i in range(agent.type):
            for j in range(agent.type):
                if agent.possibilities[
                    self.square[0] * agent.type + i, self.square[1] * agent.type + j, self.number - 1] == 1:
                    count += 1
        if count == 1:
            for i in range(agent.type):
                for j in range(agent.type):
                    if agent.possibilities[
                        self.square[0] * agent.type + i, self.square[1] * agent.type + j, self.number - 1] == 1 and \
                            agent.grid[self.square[0] * agent.type + i][self.square[1] * agent.type + j] == 0:
                        agent.grid[self.square[0] * agent.type + i][self.square[1] * agent.type + j] = self.number
                        agent.update_possibilities(self.number,
                                                   (self.square[0] * agent.type + i, self.square[1] * agent.type + j))
                        self.flag = 1
        if self.square[1] == agent.type - 1:
            if self.square[0] == agent.type - 1:
                if self.number == agent.dimension:
                    self.number = -1
                else:
                    self.square = [0, 0]
                    self.number += 1
            else:
                self.square[1] = 0
                self.square[0] += 1
        else:
            self.square[1] += 1
        pass


class Possibilities_Line_State(State):
    def __init__(self):
        super().__init__("Possibilities_Line")
        self.number = 1
        self.line = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.number == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Line2_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        for square_column in range(agent.type):
            agent.line_possibility((self.line, square_column))
            number_of_possible_lines = 0
            line = 0
            for possible_lines in range(agent.type):
                if agent.possibilities_line[self.line, square_column, self.number - 1, possible_lines] == 1:
                    number_of_possible_lines += 1
                    line = possible_lines
            if number_of_possible_lines == 1:
                for column in range(agent.dimension):
                    if column // agent.type != square_column and agent.possibilities[
                        line + self.line * agent.type, column, self.number - 1] != 0:
                        agent.possibilities[line + self.line * agent.type, column, self.number - 1] = 0
                        self.flag = 1
        if self.line == agent.type - 1:
            self.line = 0
            if self.number == agent.dimension:
                self.number = -1
            else:
                self.number += 1
        else:
            self.line += 1
        pass


class Possibilities_Line2_State(State):
    def __init__(self):
        super().__init__("Possibilities_Line2")
        self.number = 1
        self.line = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.number == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Column_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        for square_column in range(agent.type):
            agent.line_possibility((self.line, square_column))
        for line in range(agent.type):
            number_of_possible_squares = 0
            square_column = 0
            for possible_square_column in range(agent.type):
                if agent.possibilities_line[self.line, possible_square_column, self.number - 1, line] == 1:
                    number_of_possible_squares += 1
                    square_column = possible_square_column
            if number_of_possible_squares == 1:
                for i in range(agent.type):
                    for j in range(agent.type):
                        if i != line and agent.possibilities[
                            i + self.line * agent.type, j + square_column * agent.type, self.number - 1] != 0:
                            agent.possibilities[
                                i + self.line * agent.type, j + square_column * agent.type, self.number - 1] = 0
                            self.flag = 1
        if self.line == agent.type - 1:
            self.line = 0
            if self.number == agent.dimension:
                self.number = -1
            else:
                self.number += 1
        else:
            self.line += 1
        pass


class Possibilities_Column_State(State):
    def __init__(self):
        super().__init__("Possibilities_Column")
        self.number = 1
        self.column = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.number == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Column2_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        for square_line in range(agent.type):
            agent.column_possibility((square_line, self.column))
            number_of_possible_columns = 0
            column = 0
            for possible_columns in range(agent.type):
                if agent.possibilities_column[square_line, self.column, self.number - 1, possible_columns] == 1:
                    number_of_possible_columns += 1
                    column = possible_columns
            if number_of_possible_columns == 1:
                for line in range(agent.dimension):
                    if line // agent.type != square_line and agent.possibilities[
                        line, column + self.column * agent.type, self.number - 1] != 0:
                        agent.possibilities[line, column + self.column * agent.type, self.number - 1] = 0
                        self.flag = 1
        if self.column == agent.type - 1:
            self.column = 0
            if self.number == agent.dimension:
                self.number = -1
            else:
                self.number += 1
        else:
            self.column += 1
        pass


class Possibilities_Column2_State(State):
    def __init__(self):
        super().__init__("Possibilities_Column2")
        self.number = 1
        self.column = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.number == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Pair_Line_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        for square_line in range(agent.type):
            agent.line_possibility((square_line, self.column))
        for column in range(agent.type):
            number_of_possible_squares = 0
            square_line = 0
            for possible_square_line in range(agent.type):
                if agent.possibilities_column[possible_square_line, self.column, self.number - 1, column] == 1:
                    number_of_possible_squares += 1
                    square_line = possible_square_line
            if number_of_possible_squares == 1:
                for i in range(agent.type):
                    for j in range(agent.type):
                        if j != column and agent.possibilities[
                            i + square_line * agent.type, j + self.column * agent.type, self.number - 1] != 0:
                            agent.possibilities[
                                i + square_line * agent.type, j + self.column * agent.type, self.number - 1] = 0
                            self.flag = 1
        if self.column == agent.type - 1:
            self.column = 0
            if self.number == agent.dimension:
                self.number = -1
            else:
                self.number += 1
        else:
            self.column += 1
        pass


class Possibilities_Pair_Line_State(State):
    def __init__(self):
        super().__init__("Possibilities_Pair_Line")
        self.number1 = 1
        self.number2 = 2
        self.line = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.line == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Pair_Line2_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        cont = 0
        aux = np.zeros(agent.dimension)
        aux[self.number1 - 1] = 1
        aux[self.number2 - 1] = 1
        for j in range(agent.dimension):
            if compare(agent.possibilities, self.line, j, aux):
                cont += 1
        if cont == 2:
            for j in range(agent.dimension):
                if not (compare(agent.possibilities, self.line, j, aux)) and (
                        agent.possibilities[self.line, j, self.number1 - 1] == 1 or agent.possibilities[
                    self.line, j, self.number2 - 1] == 1) and (
                        agent.possibilities[self.line, j, self.number1 - 1] != 0 or agent.possibilities[
                    self.line, j, self.number2 - 1] != 0):
                    agent.possibilities[self.line, j, self.number1 - 1] = 0
                    agent.possibilities[self.line, j, self.number2 - 1] = 0
                    self.flag = 1
        if self.line == agent.dimension - 1:
            if self.number1 == agent.dimension - 1:
                self.line = -1
            else:
                if self.number2 == agent.dimension:
                    self.number1 += 1
                    self.number2 = self.number1 + 1
                else:
                    self.number2 += 1
        else:
            if self.number1 == agent.dimension - 1:
                self.line += 1
                self.number1 = 1
                self.number2 = 2
            else:
                if self.number2 == agent.dimension:
                    self.number1 += 1
                    self.number2 = self.number1 + 1
                else:
                    self.number2 += 1
        pass


class Possibilities_Pair_Line2_State(State):
    def __init__(self):
        super().__init__("Possibilities_Pair_Line2")
        self.number1 = 1
        self.number2 = 2
        self.line = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.line == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Pair_Column_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        cont1 = 0
        cont2 = 0
        columns1 = 0
        columns2 = 0
        aux = np.zeros(agent.dimension)
        aux[self.number1 - 1] = 1
        aux[self.number2 - 1] = 1
        for j in range(agent.dimension):
            if agent.possibilities[self.line, j, self.number1 - 1] == 1 and agent.possibilities[
                self.line, j, self.number2 - 1] == 1:
                columns1 = columns2
                columns2 = j
                cont1 += 1
            if agent.possibilities[self.line, j, self.number1 - 1] == 1 or agent.possibilities[
                self.line, j, self.number2 - 1] == 1:
                cont2 += 1
        if cont1 == 2 and cont2 == 2 and (not (compare(agent.possibilities, self.line, columns1, aux)) or not (
        compare(agent.possibilities, self.line, columns2, aux))):
            agent.possibilities[self.line, columns1, :] = aux[:]
            agent.possibilities[self.line, columns2, :] = aux[:]
            self.flag = 1
        if self.line == agent.dimension - 1:
            if self.number1 == agent.dimension - 1:
                self.line = -1
            else:
                if self.number2 == agent.dimension:
                    self.number1 += 1
                    self.number2 = self.number1 + 1
                else:
                    self.number2 += 1
        else:
            if self.number1 == agent.dimension - 1:
                self.line += 1
                self.number1 = 1
                self.number2 = 2
            else:
                if self.number2 == agent.dimension:
                    self.number1 += 1
                    self.number2 = self.number1 + 1
                else:
                    self.number2 += 1
        pass


class Possibilities_Pair_Column_State(State):
    def __init__(self):
        super().__init__("Possibilities_Pair_Column")
        self.number1 = 1
        self.number2 = 2
        self.column = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.column == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Pair_Column2_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        cont = 0
        aux = np.zeros(agent.dimension)
        aux[self.number1 - 1] = 1
        aux[self.number2 - 1] = 1
        for i in range(agent.dimension):
            if compare(agent.possibilities, i, self.column, aux):
                cont += 1
        if cont == 2:
            for i in range(agent.dimension):
                if not (compare(agent.possibilities, i, self.column, aux)) and (
                        agent.possibilities[i, self.column, self.number1 - 1] == 1 or agent.possibilities[
                    i, self.column, self.number2 - 1] == 1) and (
                        agent.possibilities[i, self.column, self.number1 - 1] != 0 or agent.possibilities[
                    i, self.column, self.number2 - 1] != 0):
                    agent.possibilities[i, self.column, self.number1 - 1] = 0
                    agent.possibilities[i, self.column, self.number2 - 1] = 0
                    self.flag = 1
        if self.column == agent.dimension - 1:
            if self.number1 == agent.dimension - 1:
                self.column = -1
            else:
                if self.number2 == agent.dimension:
                    self.number1 += 1
                    self.number2 = self.number1 + 1
                else:
                    self.number2 += 1
        else:
            if self.number1 == agent.dimension - 1:
                self.column += 1
                self.number1 = 1
                self.number2 = 2
            else:
                if self.number2 == agent.dimension:
                    self.number1 += 1
                    self.number2 = self.number1 + 1
                else:
                    self.number2 += 1
        pass


class Possibilities_Pair_Column2_State(State):
    def __init__(self):
        super().__init__("Possibilities_Pair_Column2")
        self.number1 = 1
        self.number2 = 2
        self.column = 0
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.column == -1:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Pair_Square_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        cont1 = 0
        cont2 = 0
        line1 = 0
        line2 = 0
        aux = np.zeros(agent.dimension)
        aux[self.number1 - 1] = 1
        aux[self.number2 - 1] = 1
        for i in range(agent.dimension):
            if agent.possibilities[i, self.column, self.number1 - 1] == 1 and agent.possibilities[
                i, self.column, self.number2 - 1] == 1:
                line1 = line2
                line2 = i
                cont1 += 1
            if agent.possibilities[i, self.column, self.number1 - 1] == 1 or agent.possibilities[
                i, self.column, self.number2 - 1] == 1:
                cont2 += 1
        if cont1 == 2 and cont2 == 2 and (not (compare(agent.possibilities, line1, self.column, aux[:])) or not (
        compare(agent.possibilities, line2, self.column, aux))):
            agent.possibilities[line1, self.column, :] = aux[:]
            agent.possibilities[line2, self.column, :] = aux[:]
            self.flag = 1
        if self.column == agent.dimension - 1:
            if self.number1 == agent.dimension - 1:
                self.column = -1
            else:
                if self.number2 == agent.dimension:
                    self.number1 += 1
                    self.number2 = self.number1 + 1
                else:
                    self.number2 += 1
        else:
            if self.number1 == agent.dimension - 1:
                self.column += 1
                self.number1 = 1
                self.number2 = 2
            else:
                if self.number2 == agent.dimension:
                    self.number1 += 1
                    self.number2 = self.number1 + 1
                else:
                    self.number2 += 1
        pass


class Possibilities_Pair_Square_State(State):
    def __init__(self):
        super().__init__("Possibilities_Pair_Square")
        self.number1 = 1
        self.number2 = 2
        self.square = [0, 0]
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.square == [-1, -1]:
            if self.flag == 0:
                state_machine.change_state(Possibilities_Pair_Square2_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        cont = 0
        aux = np.zeros(agent.dimension)
        aux[self.number1 - 1] = 1
        aux[self.number2 - 1] = 1
        for i in range(agent.type):
            for j in range(agent.type):
                if compare(agent.possibilities, i + agent.type * self.square[0], j + agent.type * self.square[1], aux):
                    cont += 1
        if cont == 2:
            for i in range(agent.type):
                for j in range(agent.type):
                    position = [i + agent.type * self.square[0], j + agent.type * self.square[1]]
                    if not (compare(agent.possibilities, position[0], position[1], aux)) and (
                            agent.possibilities[position[0], position[1], self.number1 - 1] == 1 or agent.possibilities[
                        position[0], position[1], self.number2 - 1] == 1) and (
                            agent.possibilities[position[0], position[1], self.number1 - 1] != 0 or agent.possibilities[
                        position[0], position[1], self.number2 - 1] != 0):
                        agent.possibilities[position[0], position[1], self.number1 - 1] = 0
                        agent.possibilities[position[0], position[1], self.number2 - 1] = 0
                        self.flag = 1
        if self.square[0] == agent.type - 1:
            if self.square[1] == agent.type - 1:
                if self.number1 == agent.dimension - 1:
                    self.square = [-1, -1]
                else:
                    if self.number2 == agent.dimension:
                        self.number1 += 1
                        self.number2 = self.number1 + 1
                    else:
                        self.number2 += 1
            else:
                if self.number1 == agent.dimension - 1:
                    self.square[1] += 1
                    self.number1 = 1
                    self.number2 = 2
                else:
                    if self.number2 == agent.dimension:
                        self.number1 += 1
                        self.number2 = self.number1 + 1
                    else:
                        self.number2 += 1
        else:
            if self.square[1] == agent.type - 1:
                if self.number1 == agent.dimension - 1:
                    self.square[1] = 0
                    self.square[0] += 1
                    self.number1 = 1
                    self.number2 = 2
                else:
                    if self.number2 == agent.dimension:
                        self.number1 += 1
                        self.number2 = self.number1 + 1
                    else:
                        self.number2 += 1
            else:
                if self.number1 == agent.dimension - 1:
                    self.square[1] += 1
                    self.number1 = 1
                    self.number2 = 2
                else:
                    if self.number2 == agent.dimension:
                        self.number1 += 1
                        self.number2 = self.number1 + 1
                    else:
                        self.number2 += 1
        pass


class Possibilities_Pair_Square2_State(State):
    def __init__(self):
        super().__init__("Possibilities_Pair_Square2")
        self.number1 = 1
        self.number2 = 2
        self.square = [0, 0]
        self.flag = 0

    def check_transition(self, agent, state_machine):
        if self.square == [-1, -1]:
            if self.flag == 0:
                state_machine.change_state(End_State())
            else:
                state_machine.change_state(Fill_Numbers_State())
        pass

    def execute(self, agent):
        cont1 = 0
        cont2 = 0
        position1 = [0, 0]
        position2 = [0, 0]
        aux = np.zeros(agent.dimension)
        aux[self.number1 - 1] = 1
        aux[self.number2 - 1] = 1
        for i in range(agent.type):
            for j in range(agent.type):
                if agent.possibilities[
                    i + agent.type * self.square[0], j + agent.type * self.square[1], self.number1 - 1] == 1 and \
                        agent.possibilities[
                            i + agent.type * self.square[0], j + agent.type * self.square[1], self.number2 - 1] == 1:
                    position1 = np.copy(position2)
                    position2 = [i + agent.type * self.square[0], j + agent.type * self.square[1]]
                    cont1 += 1
                if agent.possibilities[
                    i + agent.type * self.square[0], j + agent.type * self.square[1], self.number1 - 1] == 1 or \
                        agent.possibilities[
                            i + agent.type * self.square[0], j + agent.type * self.square[1], self.number2 - 1] == 1:
                    cont2 += 1
        if cont1 == 2 and cont2 == 2 and (not (compare(agent.possibilities, position1[0], position1[1], aux)) or not (
        compare(agent.possibilities, position2[0], position2[1], aux[:]))):
            for i in range(agent.type):
                for j in range(agent.type):
                    agent.possibilities[position1[0], position1[1], :] = aux[:]
                    agent.possibilities[position2[0], position2[1], :] = aux[:]
                    self.flag = 1
        if self.square[0] == agent.type - 1:
            if self.square[1] == agent.type - 1:
                if self.number1 == agent.dimension - 1:
                    self.square = [-1, -1]
                else:
                    if self.number2 == agent.dimension:
                        self.number1 += 1
                        self.number2 = self.number1 + 1
                    else:
                        self.number2 += 1
            else:
                if self.number1 == agent.dimension - 1:
                    self.square[1] += 1
                    self.number1 = 1
                    self.number2 = 2
                else:
                    if self.number2 == agent.dimension:
                        self.number1 += 1
                        self.number2 = self.number1 + 1
                    else:
                        self.number2 += 1
        else:
            if self.square[1] == agent.type - 1:
                if self.number1 == agent.dimension - 1:
                    self.square[1] = 0
                    self.square[0] += 1
                    self.number1 = 1
                    self.number2 = 2
                else:
                    if self.number2 == agent.dimension:
                        self.number1 += 1
                        self.number2 = self.number1 + 1
                    else:
                        self.number2 += 1
            else:
                if self.number1 == agent.dimension - 1:
                    self.square[1] += 1
                    self.number1 = 1
                    self.number2 = 2
                else:
                    if self.number2 == agent.dimension:
                        self.number1 += 1
                        self.number2 = self.number1 + 1
                    else:
                        self.number2 += 1
        pass


class End_State(State):
    def __init__(self):
        super().__init__("End")

    def check_transition(self, agent, state_machine):
        pass

    def execute(self, agent):
        pass
