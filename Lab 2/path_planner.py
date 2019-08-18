from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # Todo: implement the Dijkstra algorithm
        self.node_grid.reset()
        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        goal_node = self.node_grid.get_node(goal_position[0], goal_position[1])
        pq = []
        heapq.heappush(pq, (start_node.f, start_node))
        start_node.f = 0
        start_node.closed = True
        node = start_node
        while node != goal_node:
            f, node = heapq.heappop(pq)
            node.closed = True
            for sucessor_tuple in self.node_grid.get_successors(node.i, node.j):
                sucessor = self.node_grid.get_node(sucessor_tuple[0], sucessor_tuple[1])
                if sucessor.f > node.f + self.cost_map.get_edge_cost((node.i, node.j), (sucessor.i, sucessor.j)) and sucessor.closed is False:
                    sucessor.f = node.f + self.cost_map.get_edge_cost((node.i, node.j), (sucessor.i, sucessor.j))
                    sucessor.parent = node
                    heapq.heappush(pq, (sucessor.f, sucessor))
        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        return self.construct_path(goal_node), goal_node.f  # Feel free to change this line of code

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # Todo: implement the Greedy Search algorithm
        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        self.node_grid.reset()
        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        goal_node = self.node_grid.get_node(goal_position[0], goal_position[1])
        pq = []
        start_node.f = start_node.distance_to(goal_node.i, goal_node.j)
        start_node.g = 0
        heapq.heappush(pq, (start_node.f, start_node))
        node = start_node
        while node != goal_node:
            f, node = heapq.heappop(pq)
            node.closed = True
            for sucessor_tuple in self.node_grid.get_successors(node.i, node.j):
                sucessor = self.node_grid.get_node(sucessor_tuple[0], sucessor_tuple[1])
                if sucessor.closed is False:
                    sucessor.parent = node
                    sucessor.g = node.g + self.cost_map.get_edge_cost((node.i, node.j), (sucessor.i, sucessor.j))
                    sucessor.f = sucessor.distance_to(goal_node.i, goal_node.j)
                    heapq.heappush(pq, (sucessor.f, sucessor))
                    sucessor.closed = True
                    if sucessor == goal_node:
                        return self.construct_path(goal_node), goal_node.g  # Feel free to change this line of code
    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # Todo: implement the A* algorithm
        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        self.node_grid.reset()
        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        goal_node = self.node_grid.get_node(goal_position[0], goal_position[1])
        pq = []
        start_node.g = 0
        start_node.f = start_node.distance_to(goal_node.i, goal_node.j)
        heapq.heappush(pq, (start_node.f, start_node))
        node = start_node
        while node != goal_node:
            f, node = heapq.heappop(pq)
            node.closed = True
            if node == goal_node:
                return self.construct_path(goal_node), goal_node.g  # Feel free to change this line of code
            for sucessor_tuple in self.node_grid.get_successors(node.i, node.j):
                sucessor = self.node_grid.get_node(sucessor_tuple[0], sucessor_tuple[1])
                if sucessor.closed is False and sucessor.f > node.g + self.cost_map.get_edge_cost((node.i, node.j), (sucessor.i, sucessor.j)) + sucessor.distance_to(goal_node.i, goal_node.j):
                    sucessor.g = node.g + self.cost_map.get_edge_cost((node.i, node.j), (sucessor.i, sucessor.j))
                    sucessor.f = sucessor.g + sucessor.distance_to(goal_node.i, goal_node.j)
                    sucessor.parent = node
                    heapq.heappush(pq, (sucessor.f, sucessor))
