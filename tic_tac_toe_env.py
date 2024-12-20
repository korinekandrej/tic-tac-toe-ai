import gymnasium as gym
from gymnasium import spaces
import itertools
import pygame
import time
import numpy as np


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=3, dimension=2, connect_n=3, render_mode=None):
        self.size = size
        self.dimension = dimension
        self.connect_n = connect_n

        # nunmber of actions: number of cells in grid
        self.action_space = spaces.Discrete(size**dimension)
        self.observation_space = spaces.Box(low=-1, high=1, shape=self._get_shape(), dtype=np.int8)
        self.grid = np.zeros(self._get_shape(), dtype=np.int8)

        self._intersecting_axes: np.array = self._get_intersecting_axes() # to only calculate it once

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window_size = 320
        self.window = None
        self.clock = None
    
    # returns grid shape as tuple
    def _get_shape(self):
        return (self.size,)*self.dimension
    
    # convert linear index to multi-dimensional coordinates
    # for converting Actions to a place on the grid
    def _index_to_coors(self, index):
        return tuple(np.unravel_index(index, self._get_shape()))
    
    def _coors_to_index(self, coors):
        return np.ravel_multi_index(coors, self._get_shape())

    # All coordinates' parts have to be from [0(included) to size)
    def _are_coors_valid(self, coors: tuple):
        coors = np.array(coors)  # Ensure it's a NumPy array
        return np.all((0 <= coors) & (coors < self.size)) #FixedError: from all to np.all (to fix ambiguity)
    
    def _is_grid_full(self):
        return np.all(self.grid != 0)
    

    ###############################
    #  Game Logic
    ###############################

    def _is_action_valid(self, action):
        coords = self._index_to_coors(action)
        return self.grid[coords] == 0
    
    # symbol_placed: -1 or 1
    def _is_game_over(self, last_action):
        last_placed_symbol = self.grid[self._index_to_coors(last_action)]

        # based on the last action of a player (so I would not need to check the whole grid)
        # looks for n same symbols on the all axes of the last placed symbol
        # an axis is represented by a vector laying on it

        axis_items: np.array
        for axis in self._intersecting_axes:
            axis_items = self._get_line(self._index_to_coors(last_action), axis)

            if self._has_n_subsequent_numbers(self.connect_n, last_placed_symbol, axis_items):
                return True

        return False
            
    
    # Works for a grid of any dimensions
    def _get_intersecting_axes(self):
        vectors = list(itertools.product([-1, 0, 1], repeat=self.dimension))
        vectors.remove((0,)*self.dimension) # remove the (0,0,0...) vector (pointing to itself)

        axes = []
        for vector in vectors:
            # Removes opposite vectors (as both vector and its opposite vector make only one axis)
            if tuple(-np.array(vector)) not in [tuple(ax) for ax in axes]:
                axes.append(np.array(vector))

        return np.array(axes)

    # Get all tiems from a line in the grid
    # The coordinates and direction should be the same size
    def _get_line(self, coors: int, direction: np.array):
        items = []
        coorsArr = np.array(coors, dtype=int)

        # Assumes the inputted coordinates are valid
        while self._are_coors_valid(tuple(coorsArr - direction)):
            coorsArr -= direction # find the coordinates at the end of the line
            # POSSIBLE IMPROVEMENT: collect items already on the way to the end of line,
            # then just continue collecting from "coors" to the start

        ##print(f"_get_line 1")
        while self._are_coors_valid(tuple(coorsArr)):
            items.append(self.grid[tuple(coorsArr)])
            coorsArr += direction # go from the end to the front of the line

        ##print(f"_get_line end")
        return np.array(items)
    

    def _has_n_subsequent_numbers(self, n: int, number: int, items: np.array):
        ##print(f"_has_n_subsequent_numbers started {n} {number} {items}")
        chain = 0
        for item in items:
            if item == number:
                chain += 1
                if chain == n:
                    return True
            else:
                chain = 0
        return False

    # Opponent of the agent
    # Returns their validated action
    def _get_action_from_opponent(self):
        # Randomly choose
        free_cells = np.argwhere(self.grid == 0)

        random_index = np.random.choice(len(free_cells)) # When all cells are filled, this will raise an error, but it should not happen
        random_cell = free_cells[random_index]

        return self._coors_to_index(tuple(random_cell))


    ###############################
    #  Other
    ###############################
    
    def _get_obs(self):
        return self.grid

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(self._get_shape(), dtype=np.int8)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        print(f"STEP START")
        
        #time.sleep(2)
        #print(f"action: {action}")
        #print(f"self._index_to_coors(action): {self._index_to_coors(action)}")
        #print(f"self._is_action_valid(action): {self._is_action_valid(action)}")

        reward = 0
        terminated = False

        # Draw (after agent's move I have to check for draw again, see lower)
        if self._is_grid_full():
            terminated = True
            reward = 0
            return self._get_obs(), reward, terminated, False, self._get_info()

        # If invalid, punish Agent and skip step (Agent will retry their move)
        if not self._is_action_valid(action):
            reward = -1
            return self._get_obs(), reward, terminated, False, self._get_info()
        
        self.grid[self._index_to_coors(action)] = 1
        # Win
        if self._is_game_over(action):
            terminated = True
            reward = 1
        else:
            if self.render_mode == "human":
                time.sleep(0.5) #wait before rendering to avoid rapid blinking
                self._render_frame()

            print(f"agent: {action}")
            print(self.grid) #Show agent's move

            # Draw (not very code efficient to check for draw second time, but it works)
            if self._is_grid_full():
                terminated = True
                reward = 0
                return self._get_obs(), reward, terminated, False, self._get_info()

            # Opponents turn
            opponents_action = self._get_action_from_opponent() # assumes it is valid (should be checked elsewhere)
            self.grid[self._index_to_coors(opponents_action)] = -1
            #Lose
            if self._is_game_over(opponents_action):
                terminated = True
                reward = -1

            print(f"opponent: {opponents_action}")
            print(self.grid) #Show opponent's move
            print('\n')
            
            if self.render_mode == "human":
                time.sleep(0.5) #wait before rendering to avoid rapid blinking
                self._render_frame()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)


        # Drawing is supported for 2D only
        if self.dimension == 2:
            for j in range(self.size):
                for i in range(self.size):
                    if self.grid[i, j] == 1:
                        pygame.draw.rect(canvas, (0, 255, 120), pygame.Rect(i * pix_square_size, j * pix_square_size, pix_square_size, pix_square_size))
                    elif self.grid[i, j] == -1:
                        pygame.draw.rect(canvas, (255, 120, 0), pygame.Rect(i * pix_square_size, j * pix_square_size, pix_square_size, pix_square_size))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
        
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()