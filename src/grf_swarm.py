#!/usr/bin/env python3

# Implementation of
#
#

from tqdm import tqdm
import numpy as np
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib

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
		self.kernel_rS = self.kernel_generation(self.rS)

		self.fig = plt.figure()
		#self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
		self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
							xlim=(0, self.WORLD), ylim=(0, self.WORLD))
		self.ax.grid(color='gray', linestyle='-', linewidth=0.1)
		self.ax = self.fig.gca()
		self.ax.set_xticks(np.arange(0, self.WORLD, 1))
		self.ax.set_yticks(np.arange(0, self.WORLD, 1))
		self.ax.set_xlim([0, self.WORLD])
		self.ax.set_yticklabels(range(self.WORLD), rotation=0, fontsize=6)
		self.ax.set_xticklabels(range(self.WORLD), rotation=90, fontsize=6)
		self.ax.set_ylim([0, self.WORLD])
		self.ax.set_xlabel("X (meters)")
		self.ax.set_ylabel("Y (meters)")

		self.ax.plot(self.x[:, 0], self.x[:, 1], 'o', color='blue', ms=5)
		plt.ion()
		plt.show()

	def kernel_generation(self, size):
		kernel = []
		size = int(size)
		for i in range(-size, size+1):
			for j in range(-size, size+1):
				kernel.append([i, j])
		return np.array(kernel)

	def neighborhood(self, xs):
		ds = np.sqrt(np.power(self.x - xs, 2).sum(axis=1))
		tau_s = self.x[(ds <= self.rI) & (ds > 0)]
		return tau_s

	def obstacles(self, xs):
		os = self.kernel_rS + xs
		ds = np.power(zs - xs, 2).sum(axis=1)
		os = os[(ds <= self.rS)]
		os = os[(os[:,0] < 0) & (os[:,0] >= self.WORLD)]
		os = os[(os[:,1] < 0) & (os[:,1] >= self.WORLD)]


	def mobility(self, xs):
		zs = self.kernel_rM + xs
		ds = np.power(zs - xs, 2).sum(axis=1)
		zs = zs[(ds <= self.rM)]
		zs = zs[(zs[:,0] >= 0) & (zs[:,0] < self.WORLD)]
		zs = zs[(zs[:,1] >= 0) & (zs[:,1] < self.WORLD)]
		return zs

	def update(self, n):
		T = self.temperature(5.0, n)
		x_new = self.x[:]
		for i in range(self.x.shape[0]): # for each robot
			xs = self.x[i,:]
			tau_s = self.neighborhood(xs)
			Fs = self.mobility(xs)

			Z = 0
			for zs in Fs: # compute denominator
				tau_s = self.neighborhood(zs)
				Hz = self.Us(zs)
				for xt in tau_s:
					Hz += self.Ust(zs, xt)
				Z += np.exp(-Hz/T)
			#print("Z:"),
			#print(Z)
			pTs = []
			for ys in Fs:
				tau_s = self.neighborhood(ys)
				Hy = self.Us(ys)
				for xt in tau_s:
					Hy += self.Ust(ys, xt)
				p = np.exp(-Hy/T)/Z
				pTs.append(p)
			#print("probs:"),
			#print(pTs)
			#print("END\n")
			pTs = np.array(pTs)
			for j in range(pTs.shape[0]):
				xs = Fs[np.random.choice(Fs.shape[0], 1, p=pTs),:]
				if np.any((x_new == xs).sum(axis=1) == 2):
					pTs[j] = 0.0
					pTs = pTs/pTs.sum()
				else:
					x_new[i,:] = xs
					break

		self.x = x_new[:]

		# Condition - Do not get out from the world
		self.x[self.x[:,0] >= self.WORLD, 0] = self.WORLD-1
		self.x[self.x[:,1] >= self.WORLD, 1] = self.WORLD-1
		self.x[self.x[:,0] < 0, 0] = 0
		self.x[self.x[:,1] < 0, 1] = 0


	def dist(self, xs, xt):
		dx = xs[0] - xt[0]
		dy = xs[1] - xt[1]
		return np.sqrt(dx*dx + dy*dy)

	def Us(self, xs):
		#return self.dist(xs, [int(self.WORLD/2.0), int(self.WORLD/2.0)])
		return 0

	def Ust(self, xs, xt):
		ust = 0
		d = self.dist(xs, xt)
		if d == 0:
			ust = 10.0
		else:
			ust = -1.0/(d)
		return ust

	def temperature(self, t0, n):
		return t0/(np.log(n))


	def display(self):
		self.ax.lines = []
		figW, figH = self.ax.get_figure().get_size_inches()
		
		self.ax.plot(self.x[:, 0]+0.5, self.x[:, 1]+0.5, 'o', color='blue', ms=5*figH/4.8)
		self.ax.plot(self.x[:, 0]+0.5, self.x[:, 1]+0.5, 'o', color='blue', ms=2*self.rI*figH/4.8, fillstyle='none', alpha=0.1) # fig.dpi/72.
		plt.pause(0.0001)
		plt.draw()
		plt.pause(0.0001)


	def screenshot(self, filename):
		plt.savefig(filename, dpi=500)

ITERATIONS = 650

if __name__ == '__main__':
	gs = GibbsSwarm(ROBOTS=40, WORLD=50, rS=13*np.sqrt(2)+2, rI=13*np.sqrt(2), rM=1, seed=100)
	print("Starting")
	for i in tqdm(range(ITERATIONS), ncols=100):
		gs.update(i+1)
		if i % 1 == 0:
			gs.display()
	input("Press the <ENTER> key to finish...")
	plt.close('all')