import sys
from collections import deque
import heapq, itertools



class PuzzleState:
    """
       Represents a state of the sliding puzzle.
       """
    def __init__(self, tiles, parent=None, move=None, g_score=0):
        self.tiles = tiles
        self.parent = parent
        self.move = move
        self.empty_pos = self.tiles.index(0)
        self.dimension = int(len(tiles) ** 0.5)
        self.g_score = g_score  # Cost to reach this state

    def get_moves(self):
        """
        Generates the possible moves of the empty space (0).
        """
        moves = []
        row, col = divmod(self.empty_pos, self.dimension)
        if row > 0: moves.append(-self.dimension)  # Up
        if row < self.dimension - 1: moves.append(self.dimension)  # Down
        if col > 0: moves.append(-1)  # Left
        if col < self.dimension - 1: moves.append(1)  # Right
        return moves

    def generate_new_state(self, move):
        """
        Generates a new state by moving a tile into the empty space.
        """
        new_tiles = self.tiles[:]  # Create a copy of the current tile arrangement
        index = self.empty_pos + move # Calculate the index of the tile to be moved
        moved_tile = new_tiles[index] # Identify the tile that will move into the empty space
        new_tiles[self.empty_pos], new_tiles[index] = new_tiles[index], new_tiles[self.empty_pos]
        # Swap the identified tile with the empty space to create the new tile arrangement
        new_g_score = self.g_score + 1  # Increment the g-score for the new state
        return PuzzleState(new_tiles, self, moved_tile, new_g_score)  # Return a new PuzzleState instance with the updated tiles
    def is_goal(self):
        """
        Checks if the current state is the goal state.
        """
        return self.tiles == list(range(len(self.tiles)))

    def row_col_heuristic(self):
        """
        Calculates a heuristic based on whether tiles are in their correct row or column.
        Adds 1 for each dimension the tile is misplaced.
        """
        total_cost = 0
        for i, tile in enumerate(self.tiles):
            if tile != 0:  # Ignore the empty space
                # Calculate the current row and column for the tile
                current_row, current_col = divmod(i, self.dimension)
                # Calculate the goal row and column for the tile
                goal_position = tile - 1  # Subtract 1 because tiles are numbered from 1 to 8
                goal_row, goal_col = divmod(goal_position, self.dimension)

                # Check if the tile is not in the correct row
                if current_row != goal_row:
                    total_cost += 1

                # Check if the tile is not in the correct column
                if current_col != goal_col:
                    total_cost += 1

        return total_cost

    def __str__(self):
        return ' '.join(map(str, self.tiles))
def bfs(initial_state):
    """
    Performs Breadth-First Search to find the path to the goal state.
    """
    visited = set() # Set to keep track of visited states to avoid revisiting them
    queue = deque([initial_state]) # Queue to manage the states to be explored in FIFO order
    while queue:
        current_state = queue.popleft() # Get the next state to explore from the queue
        if current_state.is_goal():
            return current_state  # If the goal state is found, return it
        visited.add(str(current_state)) # Mark the current state as visited
        for move in current_state.get_moves():
            next_state = current_state.generate_new_state(move)
            if str(next_state) not in visited:  # If the generated state has not been visited, add it to the queue
                queue.append(next_state)
    return None

def dls(current_state, depth, visited):
    """
    Performs Depth-Limited Search for IDDFS.
    """
    if depth == 0 and current_state.is_goal():
        return current_state

    # If there is still depth remaining, explore further
    if depth > 0:
        visited.add(str(current_state))  # Mark the current state as visited
        for move in current_state.get_moves():
            next_state = current_state.generate_new_state(move)
            if str(next_state) not in visited:
                found = dls(next_state, depth - 1, visited)
                if found:
                    return found  # If the goal state is found in a deeper level, return it
        visited.remove(str(current_state))  # Remove the current state from visited after exploring its children
    return None

def iddfs(initial_state, max_depth):
    """
    Combines the space-efficiency of Depth-First Search (DFS) with the optimal
    solution guarantee of Breadth-First Search (BFS)
    """
    for depth in range(max_depth):
        visited = set() # Set to keep track of visited states for each depth level
        found = dls(initial_state, depth, visited)
        if found:
            return found
    return None
def gbfs(initial_state):
    """
    Performs Greedy Best-First Search to find the path to the goal state.
    """
    visited = set()
    counter = itertools.count()  # Unique sequence count for tie-breaking in the priority queue
    priority_queue = [(initial_state.row_col_heuristic(), next(counter), initial_state)]
    while priority_queue:
        _, _, current_state = heapq.heappop(priority_queue)
        if current_state.is_goal():
            return current_state
        visited.add(str(current_state))
        # Generate all possible successor states from the current state
        for move in current_state.get_moves():
            next_state = current_state.generate_new_state(move)
            if str(next_state) not in visited:
                heapq.heappush(priority_queue, (next_state.row_col_heuristic(), next(counter), next_state))
    return None

def a_star(initial_state):
    """
    Performs the A* Search algorithm to find the shortest path to the goal state.
    Uses a priority queue to dynamically select the next best state to explore
    based on the combined cost of the path taken to reach the current state (g-score)
    and the estimated cost to reach the goal from the current state (h-score).
    """
    visited = set()  # Set to keep track of visited states to avoid cycles.
    counter = itertools.count()  # Unique sequence count for tie-breaking in the priority queue.
    # Initialize the priority queue with the initial state and its f-score.
    priority_queue = [(initial_state.row_col_heuristic(), next(counter), initial_state)]

    while priority_queue:
        # Pop the state with the lowest f-score from the priority queue.
        _, _, current_state = heapq.heappop(priority_queue)
        # If the goal state is found, return it.
        if current_state.is_goal():
            return current_state
        # Add the current state to the visited set.
        visited.add(str(current_state))

        # Generate all possible successor states from the current state.
        for move in current_state.get_moves():
            next_state = current_state.generate_new_state(move)
            # If the generated state has not been visited, calculate its f-score and add it to the priority queue.
            if str(next_state) not in visited:
                f_score = next_state.g_score + next_state.row_col_heuristic()
                heapq.heappush(priority_queue, (f_score, next(counter), next_state))

    # If the priority queue is empty and no solution was found, return None.
    return None

def reconstruct_path(goal_state):
    """
    Reconstructs the path to the goal state from the final state.
    """
    path = []
    current_state = goal_state
    while current_state.parent:  # We don't include the initial state's move (which is None)
        path.append(current_state.move)  # Add the tile that moved into the empty space
        current_state = current_state.parent
    return path[::-1]  # Reverse the path to get the correct order
def main():
    if len(sys.argv) != 10:
        print("Usage: python Tiles.py 0 1 2 3 4 5 6 7 8")
        return
    initial_tiles = list(map(int, sys.argv[1:]))
    initial_state = PuzzleState(initial_tiles)

    # Run BFS
    print("BFS")
    goal_state_bfs = bfs(initial_state)
    if goal_state_bfs:
        path_bfs = reconstruct_path(goal_state_bfs)
        print(len(path_bfs))
        print(path_bfs)
    else:
        print("No solution found.")

    # Run IDDFS
    print("\nIDDFS")
    max_depth = 50  # You can adjust the maximum depth as needed
    goal_state_iddfs = iddfs(initial_state, max_depth)
    if goal_state_iddfs:
        path_iddfs = reconstruct_path(goal_state_iddfs)
        print(len(path_iddfs))
        print(path_iddfs)
    else:
        print("No solution found.")

        # Run GBFS
    print("\nGBFS")
    goal_state_gbfs = gbfs(initial_state)
    if goal_state_gbfs:
        path_gbfs = reconstruct_path(goal_state_gbfs)
        print(len(path_gbfs))
        print(path_gbfs)
    else:
        print("No solution found.")

        # Run A*
    print("\nA*")
    goal_state_a_star = a_star(initial_state)
    if goal_state_a_star:
        path_a_star = reconstruct_path(goal_state_a_star)
        print(len(path_a_star))
        print(path_a_star)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()