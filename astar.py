import time
import pygame
import numpy as np
from collections import deque
import heapq as heap

class Node():
    """
    Node class for each cell in the A* algorithm
    """
    def __init__(self, position: tuple, parent, is_coloured):
        self.parent = parent
        self.position = position
        self.is_coloured = is_coloured

    # allows for efficient search through closed position set
    def __hash__(self): 
        return hash(self.position)
    
    # allows for the Nodes to be comparable to each other for the python heap structure
    def __lt__(self, other):
        return self.f_cost() < other.f_cost() if self.f_cost() != other.f_cost() else self.h_cost < other.h_cost

    g_cost = 0
    h_cost = 0

    def f_cost(self):
        return self.g_cost + self.h_cost
    
    # check if new position is within grid
    def in_grid(self, new_position: tuple) -> bool:
        return (0 <= self.position[0] + new_position[0] < len(grid[0]) and 
                0 <= self.position[1] + new_position[1] < len(grid))


def get_neighbours(current: Node) -> list[Node]:
    """
    Return all the valid neighbours of the current node

    Neighbours on blocks or outside of grid are invalid
    """
    # list of all neighbours, visually displayed in their relative position to the current node
    neighbour_pos = [(-1, -1), (-1, 0), (-1, 1), # [(row, col)]
                     ( 0, -1),          ( 0, 1),
                     ( 1, -1), ( 1, 0), ( 1, 1)]
    neighbours = []

    for i in range(len(neighbour_pos)):
        # exclude wall neighbours and neighbours outside of boundaries
        if current.in_grid(neighbour_pos[i]) and list(current.position + np.array(neighbour_pos[i])) not in blocks[:]:
            neighbour = Node(position=tuple(current.position + np.array(neighbour_pos[i])), parent=None, is_coloured=False)
            neighbours.append(neighbour)

    return neighbours


def get_distance(pos_A: Node, pos_B: Node) -> float:
    """
    Returns the grid distance between any 2 positions on the grid

    Assumes distance between each node is 1 unit and the diagonal distance is thus square-root(2)
    """
    y = np.abs(pos_A.position[0] - pos_B.position[0]) # rows
    x = np.abs(pos_A.position[1] - pos_B.position[1]) # columns
    return np.sqrt(2) * y + (x - y) if x > y else np.sqrt(2) * x + (y - x)


def retrace_path(start: Node, end: Node) -> list[Node]:
    """
    Returns the final calculated path of Nodes from the end Nodes to the start Nodes

    Utilizes the parent object of the Node class
    """
    final_path = []
    current = end

    # move through the node tree backwards to retrace the optimal calculated path
    while current is not start:
        # colour the final path
        if current.position != end.position:
            draw_node(current.position[0], current.position[1],  solution_colour) 

        final_path.append(current)
        # move up from the child node to the optimal parent node
        current = current.parent

    # reverse the path to be read from beginning position to end position
    final_path.reverse()
    return final_path
    

def astar(start: Node, target: Node):
    """
    Returns the shortest path of a given grid from the start Node to the target Node if path exists using A* algorithm

    Returns None if no path exists
    """
    open = [] # list of Nodes to be evaluated
    closed = set() # set of nodes already evaluated

    # initalize a heap data structure for the open set of nodes to check
    heap.heappush(open, start)

    while len(open) > 0:
        # pop the lowest costing node from the open set (always is the first element in a heap)
        current: Node = heap.heappop(open)
        closed.add(current.position)

        # path found
        if current.position == target.position: 
            print("Path found")
            path = retrace_path(start, current)
            cost = current.f_cost()

            return path, cost

        neighbours = get_neighbours(current) # get all valid neighbours

        # evaluate which neighbours of the current node to pass into the open list to be checked
        for neighbour in neighbours:
            if neighbour.position in closed or neighbour.is_coloured: # if this neighbour has been checked, move onto the next
                # colour the neighbour if it hasn't already been coloured
                if neighbour.position != start.position and neighbour.position != target.position and not neighbour.is_coloured:
                    draw_node(neighbour.position[0], neighbour.position[1], closed_node_colour)
                    neighbour.is_coloured = True
                continue 

            # if new neighbour cost value is smaller than previous neighbour cost value then update neighbour cost
            new_neighbour_cost = current.g_cost + get_distance(current, neighbour)

            if new_neighbour_cost < neighbour.g_cost or neighbour not in open:
                # set new f_cost of neighbour
                neighbour.g_cost = new_neighbour_cost
                neighbour.h_cost = get_distance(neighbour, target)
                neighbour.parent = current # set parent of neighbour to current

                # if neighbour is yet to be checked, add it to the open set to be checked later
                if neighbour.position not in open:
                    heap.heappush(open, neighbour)

                    # ensure the visualizer doesn't draw over top the start and end node
                    if neighbour.position != start and neighbour.position != target:
                        draw_node(neighbour.position[0], neighbour.position[1], neighbour_colour)
                        neighbour.is_coloured = True
            
    
    # path not found
    print("Path not found")
    return None, None


def bfs(start, end):
    """
    Breadth-First Search (BFS) for pathfinding.

    :param grid: The grid representation of the map.
    :param start: Tuple, the starting position (row, col).
    :param end: Tuple, the target position (row, col).
    :return: List of nodes representing the path from start to end
    """
    queue = deque([Node(start, None, False)])
    visited = {start}
    while queue:
        current = queue.popleft()

        # Path is found, return it
        if current.position == end:
            print("Path found")
            return retrace_path(None, current), None

        for direction in [(-1, -1), (-1, 0), (-1, 1), ( 0, -1), ( 0, 1), ( 1, -1), ( 1, 0), ( 1, 1)]:  # 8-way connectivity
            row, col = ( current.position[0] + direction[0],
                         current.position[1] + direction[1], )
            if (0 <= row < len(grid[0]) and 0 <= col < len(grid) and grid[col][row] == 0 and (col, row) not in visited ):
                visited.add((col, row))
                if (row, col) != start and (row, col) != end:
                    draw_node(row, col, neighbour_colour)
                queue.append(Node((row, col), current, False))
    print("Path not found")
    return None, None


def init_game() -> None:
    """
    Initialize the game screen
    """
    window.fill(background_colour)
    init_grid(grid_x, grid_y, space, node_size)

    return


def init_grid(grid_x: int, grid_y: int, space: int, node_size: int) -> None:
    """ 
    Initialize the grid
    """
    for y in range(grid_y):
        for x in range(grid_x):
            rect = pygame.Rect(int(x * space * node_size), int(y * space * node_size), node_size, node_size)
            pygame.draw.rect(window, node_colour, rect, 0)


def draw_node(x, y, colour) -> None:
    """ 
    Draws the given coordinates in the grid on the screen
    """
    pixel_x = x * node_size * space
    pixel_y = y * node_size * space

    rect = pygame.Rect(pixel_x, pixel_y, node_size, node_size)
    pygame.draw.rect(window, colour, rect, 0)
    pygame.display.flip()  # Partially updates screen 


def run(): 
    # initialize the starting nodes
    draw_node(start[0], start[1], target_colour)
    draw_node(target[0], target[1], start_colour)  

    # get and draw the boundaries from the grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                draw_node(j, i, block_colour)
                blocks.append([j, i])

    # initialize the start and target nodes and validate them
    Target = Node(target, None, True)
    Start = Node(start, None, True)
    Start.h_cost = get_distance(Start, Target)
    if not Start.in_grid((0,0)) or not Target.in_grid((0,0)):
        print("Invalid start/target positions")
        return
    
    algorithm = input("Choose A star (A*) or Breadth-First Search (BFS) (case-sensitive): ")
    print("Click on the visualization tab and press *SPACE*")

    # Run until the user clicks *X* to quit
    while True:
        pygame.display.set_caption(f"{algorithm} Pathfinding Visualization: Press *SPACE* to begin the algorithm")
        for event in pygame.event.get():  # Has the user clicked the close button
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN: # check if user is pressing button

                if event.key == pygame.K_SPACE:  # find path between target and start node
                    print("Running", algorithm, "Algorithm...")

                    path, cost = astar(Start, Target) if algorithm == "A*" else bfs(start, target)
        
                    if path != None:
                        print("Final Path:")
                        for node in path:
                            print(node.position)
                        print("Final cost:", cost)
        pygame.display.flip()  # Partially update screen


def main():
    init_game()
    run()

    return


if __name__ == '__main__':
    # global constants and other primative checks

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (252, 219, 3)
    PURPLE = (177, 3, 252)
    GREY_GREEN = (161, 209, 172)


    # random open grid
    open_grid =  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],]
    start_open = (7, 5)
    target_open = (30, 36)

    # human face
    face_grid = np.load("human_face.npy")
    start_face = (70, 115)
    target_face = (80, 20)

    # Subway:
    map_grid = np.load("tokyo_subway_map.npy")
    start_map = (4, 2)
    target_map = (137, 63)

    print("\nPlease choose a grid")
    grid_choice = input("Open Example (1)   |   Tokyo Subway Map (2)    |   Human Face (3) \n")
    match grid_choice:
        case "1":
            grid = open_grid
            start = start_open
            target = target_open
        case "2":
            grid = map_grid
            start = start_map
            target = target_map
        case "3":
            grid = face_grid
            start = start_face
            target = target_face

    
    grid_x = len(grid[0])
    grid_y = len(grid)
    blocks = []

    node_size = 4
    space = 1

    background_colour = (BLACK)
    node_colour = (WHITE)
    neighbour_colour = (GREEN)
    closed_node_colour = (GREY_GREEN)
    solution_colour = (YELLOW)
    block_colour = (BLACK)
    start_colour = (BLUE)
    target_colour = (RED)

    window_width = int(grid_x * node_size * space)
    window_height = int(grid_y * node_size * space)
    window = pygame.display.set_mode((len(grid[0]) * node_size, len(grid) * node_size + 40))

    main()
