"""
this generator from https://github.com/zuoxingdong/gym-maze
"""
from env_maze2.gym_maze.envs.maze import MazeEnv
from env_maze2.gym_maze.envs.generators import *

def random_maze(width:int, height:int, complexity:float, density:float, render_trace:bool):
	maze = RandomMazeGenerator(width=width, height=height, complexity=complexity, density=density)
	env = MazeEnv(maze, render_trace=render_trace)
	return env
