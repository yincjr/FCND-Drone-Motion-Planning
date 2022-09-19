from enum import Enum
from queue import PriorityQueue
from turtle import distance, st
import numpy as np
from bresenham import bresenham
from scipy.spatial import Voronoi
from collections import defaultdict
import math


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

    return grid, int(north_min), int(east_min)


def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the drone's altitude.
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3])) 
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4])) 
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can 
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min))) 
    east_size = int(np.ceil((east_max - east_min)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid 
    north_min_center = np.min(data[:, 0]) 
    east_min_center = np.min(data[:, 1])

    # return neigbhours
    adjacentNodes = defaultdict(list)

    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles 
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(north - d_north - safety_distance - north_min_center), 
                int(north + d_north + safety_distance - north_min_center), 
                int(east - d_east - safety_distance - east_min_center), 
                int(east + d_east + safety_distance - east_min_center),
            ]
            grid[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3]] = 1
            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # TODO: create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)

    # TODO: check each edge from graph.ridge_vertices for collision 
    edges = []
    for edge in graph.ridge_vertices:
        point1 = graph.vertices[edge[0]] 
        point2 = graph.vertices[edge[1]]

        cells = list(bresenham(int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1])))
        infeasible = False

        for cell in cells:
            if np.amin(cell) < 0 or cell[0] >= grid.shape[0] or cell[1] >= grid.shape[1]:
                infeasible = True
                break
            if grid[cell[0], cell[1]] == 1:
                infeasible = True
                break
        if infeasible == False:
            point1 = (point1[0], point1[1]) 
            point2 = (point2[0], point2[1]) 
            edges.append((point1, point2))

    def generateAdjacencyLst(edges):
        for u, v in edges:
            adjacentNodes[u].append(v)
            adjacentNodes[v].append(u)
        return adjacentNodes

    return grid, edges, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @ property
    def cost(self):
        return self.value[2]

    @ property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def a_star(grid, h, start, goal):

    print('goal', goal)

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]

        # print('Visiting ', current_node)

        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]

        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                # print(branch_cost, queue_cost)

                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


# TODO: Implementing the uniform cost search algorithm
def ucs(grid, start, goal):
    print('goal', goal)
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        print('Visiting ', current_node)

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                # new cost
                new_cost = action.cost + current_cost
                if next_node not in branch or new_cost < branch[next_node][0]:
                    branch[next_node] = (new_cost, current_node, action)
                    queue.put((new_cost, next_node))

    if found:
        n = goal
        path_cost = branch[n][0]
        # path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


# TODO: Implementing the iterative deepening A* search algorithm
def iterative_astar(grid, h, start, goal):

    distance = h(start, goal)
    branch = {}
    found = False
    path = [start]

    # threshold == var
    var = distance

    while True:
        visited = {}
        queue = PriorityQueue()

        queue.put((1, start))
        visited = set(start)

        path, cost, var = iterative_astar_util(grid, queue, path, branch, start, goal, visited, h)

        if (isinstance(var, bool)):
            print('Found a path.')
            # return path, distance
            return path, cost
        elif isinstance(var, int):
            if var == -1:
                print('**********************')
                print('Failed to find a path!')
                print('**********************')


def iterative_astar_util(grid, queue, path, branch, start, goal, visited, h):

    cnt = 0
    currentDistance = -1

    while not queue.empty():
        cnt += 1
        item = queue.get()
        current_node = item[1]

        if current_node == goal:
            print('No. of nodes visited: ', cnt)

            # FOUND
            n = goal
            cost = branch[n][0]
            path.append(goal)

            while branch[n][1] != start:
                path.append(branch[n][1])
                n = branch[n][1]

            path.append(branch[n][1])

            return path, cost, True

        for action in valid_actions(grid, current_node):
            # get the tuple representation
            da = action.delta
            next_node = (current_node[0] + da[0], current_node[1] + da[1])

            branch_cost = currentDistance + action.cost
            queue_cost = branch_cost + h(next_node, goal)

            if next_node not in visited:                
                visited.add(next_node)            
                branch[next_node] = (branch_cost, current_node, action)
                queue.put((queue_cost, next_node))

    return path, queue_cost, currentDistance


# TODO: With graph
def bfs(grid, start, goal, neighbour):

    print('goal', goal)
    visited = set()
    queue = PriorityQueue()
    queue.put(start)
    visited.add(start)
    parent = dict()
    parent[start] = None

    found = False
    while not queue.empty():
        current_node = queue.get()
        if (current_node == goal):
            found = True
            print('Found a path.')
            break
        for next_node in neighbour[current_node]:
            if next_node not in visited:
                queue.put(next_node)
                parent[next_node] = current_node
                visited.add(next_node)

    path = []

    if found:
        path.append(goal)
        while parent[goal] != None:
            path.append(parent[goal])
            goal = parent[goal]
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1]


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position), ord=1)


# TODO: New Heuristic: Implementing different heuristics for A* and the A* search for traversing 3 fixed points
def heuristic(position, goal_position):
    return math.sqrt((math.pow((position[0] - goal_position[0]), 2) + math.pow((position[1] - goal_position[1]), 2)))
