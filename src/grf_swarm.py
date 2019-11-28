#!/usr/bin/env python

# Implementation of
#
#

import numpy as np
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt


np.seterr(divide='ignore', invalid='ignore')
class GibbsSwarm(object):
	"""docstring for GibbsSwarm"""
	def __init__(self, ROBOTS=5, WORLD=40, seed=100):
		self.ROBOTS = ROBOTS
		self.WORLD = WORLD
		self.seed = seed
		self.setup()


	def setup(self):
		np.random.seed(self.seed)
		self.x =  (np.random.randint(low=0, high=self.WORLD, size=(self.ROBOTS, 2))) # position
		np.random.seed(None)
		print(self.x)

		self.fig = plt.figure()
		self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
		self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
							xlim=(0, self.WORLD), ylim=(0, self.WORLD))
		self.ax.grid(color='gray', linestyle='-', linewidth=0.1)
		self.ax = self.fig.gca()
		self.ax.set_xticks(np.arange(0, self.WORLD, 1))
		self.ax.set_yticks(np.arange(0, self.WORLD, 1))
		self.ax.set_xlim([0, self.WORLD])
		self.ax.set_ylim([0, self.WORLD])
		self.ax.set_xlabel("X (meters)")
		self.ax.set_ylabel("Y (meters)")
		#plt.scatter(x, y)
		self.ax.plot(self.x[:, 0], self.x[:, 1], 'o', color='blue', ms=5)
		plt.ion()
		plt.show()


	def update(self):
		self.x[:, 0] += 1
		pass


	def display(self):
		self.ax.clear()
		#self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
		#						xlim=(0, self.WORLD), ylim=(0, self.WORLD))
		self.ax.grid(color='gray', linestyle='-', linewidth=0.1)
		self.ax = self.fig.gca()
		self.ax.set_xticks(np.arange(0, self.WORLD, 1))
		self.ax.set_yticks(np.arange(0, self.WORLD, 1))
		self.ax.set_xlim([0, self.WORLD])
		self.ax.set_ylim([0, self.WORLD])
		self.ax.set_xlabel("X (meters)")
		self.ax.set_ylabel("Y (meters)")
		#plt.tight_layout()
		
		self.ax.plot(self.x[:, 0], self.x[:, 1], 'o', color='blue', ms=5)

		plt.draw()
		plt.pause(0.0005)

	def screenshot(self, filename):
		plt.savefig(filename, dpi=500)


if __name__ == '__main__':
	gs = GibbsSwarm()
	print("Starting")
	while True:
		gs.update()
		gs.display()
		#plt.close('all')