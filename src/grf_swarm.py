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
	def __init__(self, ROBOTS=5, WORLD=40, rS=10, rI=10, rM=1, seed=100):
		self.ROBOTS = ROBOTS
		self.WORLD = WORLD
		self.seed = seed
		self.rS = rS
		self.rI = rI
		self.rM = rM
		self.setup()


	def setup(self):
		np.random.seed(self.seed)
		self.x =  (np.random.randint(low=0, high=self.WORLD, size=(self.ROBOTS, 2))) # position
		np.random.seed(None)
		self.kernel_rM = self.kernel_generation(self.rM)
		self.kernel_rI = self.kernel_generation(self.rI)

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

	def kernel_generation(self, size):
		kernel = []
		for i in range(-size, size+1):
			for j in range(-size, size+1):
				kernel.append([i, j])
		return np.array(kernel)


	def update(self):
		self.x[:, 0] += 1

		for xs in self.x:
			self.parallel_gibbs(xs, self.kernel_rM+xs, self.x)

		# Condition - Do not get out from the world
		self.x[self.x[:,0] > self.WORLD, 0] = self.WORLD
		self.x[self.x[:,1] > self.WORLD, 1] = self.WORLD
		self.x[self.x[:,0] < 0, 0] = 0
		self.x[self.x[:,1] < 0, 1] = 0

	def parallel_gibbs(self, xs, ys, x):

		pass


	def dist(self, xs, xt):
		dx = xs[0] - xt[0]
		dy = xs[1] - xt[1]
		return np.sqrt(dx*dx + dy*dy)

	def rendezvous_potential(self, xs, xt):
		us = 0 # no pre-specified gathering point
		ust = 0
		d = self.dist(xs, xt)
		if d == 0:
			ust = 10
		else:
			ust = -1/(d)
		return us, ust


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
		self.ax.plot(self.x[:, 0], self.x[:, 1], 'o', color='blue', ms=2*self.rI*4.9, fillstyle='none', alpha=0.1) # fig.dpi/72.

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
		break
		#plt.close('all')