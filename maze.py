"""
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

# generator 
from generator_maze import *

env = random_maze(width=30, height=25,
	complexity=0.6, density=0.6, render_trace=True)

# define parameters
INPUT_SIZE = env.observation_space