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
import time

np.seterr(divide='ignore', invalid='ignore')
class GibbsSwarm(object):
	"""docstring for GibbsSwarm"""
	def __init__(self, ROBOTS=5, WORLD=40, rS=10, rI=10, rM=1, t0=5, control="rendezvous", seed=100):
		self.ROBOTS = ROBOTS
		self.WORLD = WORLD
		self.seed = seed
		self.rS = rS
		self.rI = rI
		self.rM = rM
		self.t0 = t0
		self.control = control
		self.setup()


	def setup(self):
		np.random.seed(self.seed)
		self.x =  (np.random.randint(low=0, high=self.WORLD, size=(self.ROBOTS, 2))) # position
		#np.random.seed(None)
		self.kernel_rM = self.kernel_generation(self.rM)
		self.kernel_rI = self.kernel_generation(self.rI)
		self.kernel_rS = self.kernel_generation(self.rS)
		self.w_obstacles = self.obstacles_gen()
		self.Energy = []

		matplotlib.use('TkAgg')
		self.fig = plt.figure()
		#self.fig2 = plt.figure()
		#self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
		self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
							xlim=(0, self.WORLD), ylim=(0, self.WORLD))
		#self.ax2 = self.fig2.add_subplot(111, aspect='equal', autoscale_on=True)
		self.ax.grid(color='gray', linestyle='-', linewidth=0.1)
		#self.ax2.grid(color='gray')
		self.ax = self.fig.gca()
		#self.ax2 = self.fig2.gca()
		self.ax.set_xticks(np.arange(0, self.WORLD, 1))
		self.ax.set_yticks(np.arange(0, self.WORLD, 1))
		self.ax.set_xlim([0, self.WORLD])
		self.ax.set_yticklabels(range(self.WORLD), rotation=0, fontsize=6)
		self.ax.set_xticklabels(range(self.WORLD), rotation=90, fontsize=6)
		self.ax.set_ylim([0, self.WORLD])
		self.ax.set_xlabel("X (meters)")
		self.ax.set_ylabel("Y (meters)")
		self.figH = 4.8
		self.ax.plot(self.x[:, 0]+0.5, self.x[:, 1]+0.5, 'o', color='blue', ms=5)
		self.ax.plot(self.x[:, 0]+0.5, self.x[:, 1]+0.5, 'o', color='blue', ms=10*self.rI*self.figH/4.8, fillstyle='none', alpha=0.1) # fig.dpi/72.
		#self.ax2.set_xlabel("Time steps")
		#self.ax2.set_ylabel("Potential Energy")
		#self.ax2.set_xlim([0, 1001])
		plt.ion()
		plt.show()

	def obstacles_gen(self):
		obs = []
		for i in range(-1, self.WORLD+1):
			obs.append([-1, i])
			obs.append([i, -1])
			obs.append([self.WORLD, i])
			obs.append([i, self.WORLD])
		return np.array(obs)


	def kernel_generation(self, size):
		kernel = []
		size = int(size)
		for i in range(-size, size+1):
			for j in range(-size, size+1):
				kernel.append([i, j])
		return np.array(kernel)

	def neighborhood(self, i, ys, x_new):
		tmp_x = np.delete(x_new, i, axis=0)
		ds = np.sqrt(np.power(tmp_x - ys, 2).sum(axis=1))
		tau_s = tmp_x[(ds <= self.rI)]
		return tau_s

	def multidim_intersect(self, arr1, arr2):
		arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
		arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
		intersected = np.intersect1d(arr1_view, arr2_view)
		return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

	def obstacles(self, xs):
		#obs = self.kernel_rS + xs
		#obs = self.multidim_intersect(obs, self.w_obstacles)
		#return obs
		obs = self.kernel_rS + xs	
		ds = np.sqrt(np.power(obs - xs, 2).sum(axis=1))
		obs = obs[(ds <= self.rS)]	
		obs = obs[(obs[:,0] >= -1) & (obs[:,0] <= self.WORLD) & (obs[:,1] >= -1) & (obs[:,1] <= self.WORLD)]	
		obs = obs[(obs[:,0] == -1) | (obs[:,0] == self.WORLD) | (obs[:,1] == -1) | (obs[:,1] == self.WORLD)]	
		return obs

	def mobility(self, xs, x_new, i):
		tmp_x = np.delete(x_new, i, axis=0)
		zs = self.kernel_rM + xs
		ds = np.sqrt(np.power(zs - xs, 2).sum(axis=1))
		zs = zs[(ds <= self.rM)]
		zs = zs[(zs[:,0] >= 0) & (zs[:,0] < self.WORLD)]
		zs = zs[(zs[:,1] >= 0) & (zs[:,1] < self.WORLD)]
		n_zs = []
		for j in range(len(zs)):
			if np.any((zs[j] == tmp_x).sum(axis=1) == 2):
				continue
			else:
				n_zs.append(zs[j])

		return np.array(n_zs)

	def update(self, n):
		self.ax.lines = []
		self.figW, self.figH = self.ax.get_figure().get_size_inches()
		T = self.temperature(self.t0, n+1)
		#np.random.shuffle(self.x)
		x_new = np.copy(self.x)
		#self.ax.plot(self.x[:, 0]+0.5, self.x[:, 1]+0.5, 'o', color='blue', ms=5*self.figH/4.8, alpha=0.2)
		np.set_printoptions(suppress=True)
		H_T = 0
		for i in range(self.x.shape[0]): # for each robot
			xs = np.copy(x_new[i,:])
			Fs = self.mobility(xs, x_new, i)
			#self.ax.plot(self.x[i, 0]+0.5, self.x[i, 1]+0.5, 'o', color='blue', ms=10*self.rI*self.figH/4.8, fillstyle='none', alpha=0.1) # fig.dpi/72.
			#self.ax.plot(self.x[i, 0]+0.5, self.x[i, 1]+0.5, 'o', color='blue', ms=3*self.figH/4.8)
			Z = 0
			H = []
			for zs in Fs: # compute mobility probability
				tau_s = self.neighborhood(i, zs, x_new)
				m = tau_s.mean(axis=0)
				obs = self.obstacles(zs)
				Hz = self.Us(zs)
				Hobs = []
				for xt in tau_s:
					Hz += self.Ust(zs, xt, m, n)
				
				for os in obs:
					Hobs.append(self.Uso(zs, os))
				if Hobs:
					Hobs = np.array(Hobs)
					#Hz += Hobs.max()			
					#Hz += Hobs.sum()			
				H.append(np.exp(-Hz/T))
			H = np.array(H)
			Z = H.sum()
			pTs = H/Z
			pTs = np.nan_to_num(pTs)
			idx = np.random.choice(Fs.shape[0], 1, p=pTs)
			x_new[i,:] = np.copy(Fs[idx,:])
			H_T += H[idx]
				
			#self.ax.plot(x_new[i, 0]+0.5, x_new[i, 1]+0.5, 'o', color='black', ms=2*self.figH/4.8)
			#for j in range(Fs.shape[0]):
			#	xs = np.copy(Fs[j, :])
			#	self.ax.plot(xs[0]+0.5, xs[1]+0.5, 'o', color='red', ms=5*self.figH/4.8, alpha=pTs[j]*0.8)
			#print(self.x[i,:],"->", x_new[i,:], "p:", pTs[idx])
			#print(pTs)
			#print("========")
			#plt.draw()
			#plt.pause(0.0001)
			#input()
		self.Energy.append(H_T)
		self.x = np.copy(x_new)
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
		if self.control == "rendezvous":
			return 0
		elif self.control == "line":
			return 0
		elif self.control == "radial":
			return 0
		else:
			return 0

	def Ust(self, xs, xt, m, i):
		ust_ = 0
		d = self.dist(xs, xt)
		if self.control == "rendezvous":
			if d == 0:
				ust_ = 10.0
			else:
				ust_ = -1.0/(d)
		elif self.control == "line":
			if d == 0.0:
				ust_ = 0.0
			else:
				if i < 20:
					ust_ = -(np.fabs(np.inner(xt-xs, [1, 1])) / (np.sqrt(2.0) * d))
				#ust_ += (np.fabs(np.inner(xt-xs, [0, 1])) / (np.sqrt(2.0) * d))*0.75/5.0 
				#ust_ += (np.fabs(np.inner(xt-xs, [-1, 0])) / (np.sqrt(2.0) * d))*0.75/5.0
				#ust_ += (np.fabs(np.inner(xt-xs, [0, -1])) / (np.sqrt(2.0) * d))*0.75/5.0 
				#ust_ += (np.fabs(np.inner(xt-xs, [1, 0])) / (np.sqrt(2.0) * d))*0.75/5.0 
				#ust_ += -(np.fabs(np.inner(xt-xs, [-1, 1])) / (np.sqrt(2.0) * d))*1/2
				#ust_ += -(np.fabs(np.inner(xt-xs, [1, -1])) / (np.sqrt(2.0) * d))*1/2
				else:
					ust_ += (np.fabs(np.inner(xt-xs, [-1, 1])) / (np.sqrt(2.0) * d))*1/2
					ust_ += (np.fabs(np.inner(xt-xs, [1, -1])) / (np.sqrt(2.0) * d))*1/2

		elif self.control == "radial":
			dm = self.dist(m, xs)*2
			if d == 0 or dm == 0:
				ust_ = 0.0
			else:
				ust_ = -1.0/(d) + 1/dm
		else:
			ust_ = 0
		return ust_

	def Uso(self, xs, xo):
		d = self.dist(xs, xo)
		if d >= int(self.rS):
			return 0.0
		else:
			#return -1.0/(1.0 - d/self.rS)
			return 1.0/(d)

	def temperature(self, t0, n):
		return t0/(np.log(n))


	def display(self):
		self.ax.lines = []
		#self.ax2.lines = []
		self.figW, self.figH = self.ax.get_figure().get_size_inches()
		
		self.ax.plot(self.x[:, 0]+0.5, self.x[:, 1]+0.5, 'o', color='blue', ms=5*self.figH/4.8)
		self.ax.plot(self.x[:, 0]+0.5, self.x[:, 1]+0.5, 'o', color='blue', ms=10*self.rI*self.figH/4.8, fillstyle='none', alpha=0.1) # fig.dpi/72.
		#self.ax2.plot(range(len(self.Energy)), self.Energy, '-', color='blue', label="Potential Energy")
		plt.draw()
		plt.pause(0.0001)


	def screenshot(self, filename):
		plt.savefig(filename, dpi=500)

ITERATIONS = 1000

if __name__ == '__main__':
	gs = GibbsSwarm(ROBOTS=40, WORLD=50, rS=13*np.sqrt(2)+2, rI=13*np.sqrt(2), rM=2, t0=5, control="rendezvous", seed=100)
	#gs = GibbsSwarm(ROBOTS=50, WORLD=50, rS=13*np.sqrt(2)+3, rI=13*np.sqrt(2), rM=3, t0=1, control="line", seed=110)

	#gs = GibbsSwarm(ROBOTS=40, WORLD=50, rS=13*np.sqrt(2)+2, rI=13*np.sqrt(2), rM=2, t0=5, control="radial", seed=100)
	#gs = GibbsSwarm(ROBOTS=40, WORLD=50, rS=8, rI=5, rM=1, seed=100)
	print("Starting")
	for i in tqdm(range(ITERATIONS), ncols=100):
		gs.update(i+1)
		gs.display()
		#gs.screenshot("data/image_"+str(i)+".png")
	input("Press the <ENTER> key to finish...")
	plt.close('all')