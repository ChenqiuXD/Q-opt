"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


def state2idx(state):
    """ This function transform state [upper_left_coor, lower_right_coor] to an index """
    if type(state)==list:
        a = [ (state[0]-5)/40, (state[1]-5)/40 ]
        idx = int( a[1] * MAZE_W + a[0])
    else:
        if state.endswith('good'):
            return 2 + 4*2
        elif state.endswith('bad1'):
            return 1 + 4*1
        elif state.endswith('bad2'):
            return 1 + 4*2
    return int(idx)

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_states = MAZE_H * MAZE_W
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 5
            done = True
            s_ = 'terminal_good'
        elif s_ == self.canvas.coords(self.hell1):
            reward = -1
            done = True
            s_ = 'terminal_bad1'
        elif s_ == self.canvas.coords(self.hell2):
            reward = -1
            done = True
            s_ = 'terminal_bad2'
        else:
            reward = -0.1
            done = False

        return reward, s_, done

    def render(self):
        time.sleep(0.1)
        self.update()

    def calc_Q_opt(self, g):
        real_Q = np.zeros([MAZE_H, MAZE_W, 4])  # 4: in sequence of [upper, lower, left, right]
        real_Q[0] = np.array([ [g**6, g**5, g**6, g**5],
                               [g**5, g**6, g**6, g**4],
                               [g**4, -1, g**5, g**3],
                               [g**3, g**2, g**4, g**3] ])
        real_Q[1] = np.array( [ [g**6, g**4, g**5, g**6],
                                [g**5, -1, g**5, -1],
                                [0, 0, 0, 0],
                                [g**3, g, -1, g**2] ] )
        real_Q[2] = np.array([ [g**5, g**3, g**4, -1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [g**2, g**2, 1, g] ])
        real_Q[3] = np.array([ [g**4, g**3, g**3, g**2],
                               [-1, g**2, g**3, g],
                               [+1, g, g**2, g**2],
                               [g, g**2, g, g**2]  ])
        return real_Q.flatten()

    def evaluate(self, agent):
        sum_reward = 0
        state = state2idx(self.reset())
        repeat_time = 3
        for _ in range(repeat_time):
            for __ in range(20): # Longest step : 20
                action = agent.choose_action(state, epsilon=0.0)
                r, s_, done=self.step(action)
                sum_reward += r
                state = state2idx(s_)

                if done:
                    break
        return sum_reward / repeat_time
        



def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = np.random.choice(4, size=1).astype(int)
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()