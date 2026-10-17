import math
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.animation as animation
import statistics as s
import seaborn as sns
import argparse
import csv
import os 
from collections import Counter



class GameOfLifeLattice:
    def __init__(self, length, width,seed):
        """Defines the game of life lattice, inputs dimensions of length and width, as well as the seed which defines which end state it will end in"""
        self.length = length
        self.width = width
        self.seed = seed
        self.dist_old = 0 #Will be used for unravelling the distance and velocity graphs
        self.dist_abs = 0
        if self.seed ==0:
            self.lattice = np.random.choice([0,1], (self.length, self.width)) #let 1 denote a live cell
        if self.seed ==1:  #Creates a block of states which do not evolve over time --> absorbing state      
            self.lattice = np.zeros((self.length, self.width))
            # 2x2 block - simplest still life
            cx, cy = self.length//2, self.width//2
            self.lattice[cx:cx+2, cy:cy+2] = 1
        if self.seed ==2:  #Creates blinkers --> Dynamic equilibrium
            self.lattice = np.zeros((self.length, self.width))
            # Place several blinkers across the lattice
            blinker_positions = [(10,10), (10,30), (30,10), (30,30), (20,20)]
            for (r, c) in blinker_positions:
                self.lattice[r, c:c+3] = 1  # horizontal blinker
        if self.seed ==3:
            selected_points = np.array([(25,3),(25,4),(25,5),(25,6),(26,6),(27,6),(28,5),(28,2),(26,2)])#Selects points corresponding to a medium spaceship starting on the LHS

            self.lattice = np.zeros((self.length,self.width))
            self.lattice[selected_points[:,0],selected_points[:,1]]=1
            #selected_points [:,0] grabs entire first column of points
            #selected_points[:,1] grabs second column

            #Now need to calculate initial position to gauge distance from position
            x_CoM = np.sum(selected_points[:,1])/len(selected_points)
            y_CoM = np.sum(selected_points[:,0])/len(selected_points)
            self.inititial_CoM = [y_CoM, x_CoM]
            self.dist_list = []

        self.livecount = np.sum(self.lattice)    
    def next_step(self): #Roll method --> Should handle periodic boundary conditions  
        """Calculates the sum of all 8 neighbor points at each individual point in the lattice using numpy roll technique
        Also generates the next lattice using criteria outlined in lecture notes"""

        Neighbor_lattice = (np.roll(self.lattice, 1, axis = 0) 
                           + np.roll(self.lattice, -1, axis = 0)
                           + np.roll(self.lattice, 1, axis = 1)
                           +np.roll(self.lattice, -1, axis = 1)
                            #4 main directions i.e., North South East and West
                            +np.roll(np.roll(self.lattice, 1, axis = 1), 1, axis = 0)  #Diagonal SouthEast
                            +np.roll(np.roll(self.lattice, 1, axis = 1), -1, axis = 0) #Diagnonal NorthEast
                            +np.roll(np.roll(self.lattice, -1, axis = 1), 1, axis = 0) #Diagonal SouthWest
                            +np.roll(np.roll(self.lattice, -1, axis = 1), -1, axis = 0) #Diagonal SouthEast
        )
        #Neighbor lattice describes the number of living cells neighboring each point in the self.lattice
        living_mask = ((self.lattice==1) & (Neighbor_lattice==2))|(Neighbor_lattice==3) #To make or statemennts in numpy need | !!!!
        #Should create a list of 1s where the cells are alive and neighboring counts are either 2 or 3, while putting zeros in all places where that is not the case
        dead_mask = (self.lattice == 0) & (Neighbor_lattice==3)
        #Creates an array of 1s for dead cells in lattice with 3 neighbors, all other values are 0

        #now updating lattice
        self.lattice = (living_mask|dead_mask).astype(int)  #(|) creates true/false array if either of conditions (i.e., masks) are met, .astype(int) converts to binary logic
        self.livecount=np.sum(self.lattice)

    def RunNoAnim(self):
        self.next_step()

    def gen_fig(self):
        """Generates the initial figure for the animation"""
        self.fig, self.ax = plt.subplots(figsize = (8,6))
        self.im = self.ax.imshow(self.lattice, cmap='binary', aspect='auto')
        self.ax.set_title('Game of life Lattice')
        plt.colorbar(self.im, ax=self.ax)

    def update(self, frame):
        """Updates the frames for each step of the animation"""
        self.next_step()
        self.im.set_data(self.lattice)
        #print(f"The number of living cells at next state is {self.livecount}")
        return self.im
    def animate_fig(self,n_iter):
        """Creates an animation of the Game of Life lattice"""
        self.n_iter = n_iter
        self.gen_fig()
        anim =animation.FuncAnimation(self.fig, self.update, self.n_iter, repeat = False, interval = 50)
        return anim
    def centerofmass(self):
        """Calculates the distance of the center of mass of the glider from the initial center of mass"""

        
        y_list, x_list = np.where(self.lattice == 1)
        #this should account for periodic boundary conditions (I hope)
        # dist = math.sqrt(np.mean(y_list)**2 + np.mean(x_list)**2)



        dy = np.abs(y_list - self.inititial_CoM[0])
        dx = np.abs(x_list - self.inititial_CoM[1])

        # periodic boundaries
        dy = np.minimum(dy, self.length - dy)
        dx = np.minimum(dx, self.width - dx)

        self.dist = np.sqrt(np.mean(dx)**2 + np.mean(dy)**2)
        if (self.dist_old - self.dist) > min(self.length, self.width) / 2: #we have wrapped around due to periodic boundary conditioning, need to handle carefully
            #Simply don't append any values for this stage
            self.dist_old = self.dist
        else:
            self.dist_abs+=abs(self.dist-self.dist_old) #Incrementally increase the absolute distance from the original position by the change in distance
            self.dist_list.append(self.dist_abs)
            self.dist_old = self.dist
    def update_glider(self, frame):
        self.next_step()
        self.centerofmass()
        self.im.set_data(self.lattice)
        #print(f"The number of living cells at next state is {self.livecount}")
        return self.im
    def animate_fig_glider(self,n_iter):
        self.n_iter = n_iter
        self.gen_fig()
        anim =animation.FuncAnimation(self.fig, self.update_glider, self.n_iter, repeat = False, interval = 50)
        return anim
    def convergence_steps(self):
        """Generates lattice and runs until 5 iterations have the same number of active cells, at which point the number of steps to achieve
        state is returned"""
        stop_criteria = 10
        same_state_counter = 0
        step_counter = 0
        #count_init = np.sum(self.lattice)
        up_bound = 2000
        count_init = self.livecount

        while same_state_counter < stop_criteria and step_counter < up_bound:  #State must maintain same number of active sites for 10 iterations to be considered equalized
            #Any state which takes more than 100000 steps to equilibrate is considered divergent
            self.next_step()
            count_final = self.livecount
            #print(f"The initial count was {count_init}, the final count is {count_final}")
            if count_init == count_final:
                same_state_counter+=1
            else:
                count_init = count_final
                same_state_counter=0
            step_counter+=1
        print(step_counter)
        if step_counter< up_bound:  #Have not seen any state which will converge, converge past 3000 steps
            return step_counter
        else: 
            return 0 #This is a placeholder, but consider all divergent states to diverge at 0, i.e, the state is as predictable as it can be at state 0


class DataAcquisition:
    def __init__(self, num_sims, LatticeClass, width, length):
        self.num_sims = num_sims
        self.width = width
        self.length = length
        self.converg_list = []
        self.vel_list = []
        self.dist_list = []
        self.latticetype = LatticeClass
    
    def converge_data(self):
        """Runs a number(num_sims) of GameOfLife Lattices generating a histogram of the number of steps necessary for convergence """
        for i in range(self.num_sims):
            self.lattice=self.latticetype(50,50,0)
            NumConverg= self.lattice.convergence_steps()
            if NumConverg!=0:
                self.converg_list.append(NumConverg)
            if i%10 == 0:
                print(f"On step {i}")
        
        #Finding most common count
        mode_steps = np.bincount(np.array(self.converg_list)).argmax()
        print(f'The most common number of converegence steps is {mode_steps}')
        plt.figure(figsize=(6, 4))
        plt.hist(self.converg_list, bins = int(len(self.converg_list)/10), density = True, histtype = 'stepfilled')
        plt.title("Steps Necessary for Convergence")
        plt.xlabel(f'Steps Till Convergence (10 Consecutive States With same Live Count)')
        plt.ylabel("Probability Density")
        plt.tight_layout()
        plt.show()

    def GliderVel(self,num_sweeps):
        """Plots distance from initial center of mass to position at later frame
        Also calculates and plots velocity (using simple change in position/time) against time
        """
        GliderLattice = self.latticetype(self.length,self.width, 3) #1 seed to launch the glider animation

        for _ in range(num_sweeps):
            GliderLattice.next_step()
            GliderLattice.centerofmass()
        #generating plot
        self.dist_list = GliderLattice.dist_list 
        plt.figure(figsize=(8,6))
        plt.plot(GliderLattice.dist_list)
        plt.title(f"Distance of Glider From Initial Position as Function of Time")
        plt.xlabel(f'Time (frame)')
        plt.ylabel(f'Distance from Initial Position')
        plt.show()
        self.vel_list = []
        for i in range(len(self.dist_list)-1):
            self.vel_list.append(abs(self.dist_list[i+1]-self.dist_list[i]))
        filtered_vel = np.array(self.vel_list, dtype = float)
        vel_mean = np.mean(filtered_vel)
        std_vel = np.std(filtered_vel)
        filtered_vel[filtered_vel<0.45] = np.nan #Every velocity value below 0.43 is not regular oscillation and stems from periodic boundary
        plt.figure(figsize=(8,6))
        plt.plot(filtered_vel[::10]) #Gets every 5th value due to the massive oscllations in the light spaceship
        plt.title(f"Velocity of Glider Versus Time")
        plt.xlabel(f'Time (frame)')
        plt.ylabel(f'Velocity of Glider at Frame')
        plt.show()

    def save_data(self, filename):
        """Save all collected data to a file."""
        with open(filename + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['converg_list', 'dist_list', 'vel_list'])
            all_lists = [self.converg_list, self.dist_list, self.vel_list]
            max_len = max(len(l) for l in all_lists)
            for i in range(max_len):
                writer.writerow([l[i] if i < len(l) else '' for l in all_lists])
        print(f"Data saved to {os.path.abspath(filename + '.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Width', type = int, help='Width of the created lattice', default  = 50)
    parser.add_argument('--Length', type = int, help='Length of the created lattice', default  = 50)
    parser.add_argument('--Seed', type = int, help='Seed of the initial state, 0 is fully random, 3 is glider state', default  = 0)
    parser.add_argument('--Anim', type = bool, help='Animate the lattice dynamics?', default  = False)
    parser.add_argument('--Data', type = int, help='The desired data to be acquired and plotted' \
    '0 = None, 1 = # steps for convergence in game of life, 2 = The GameOfLife Glider Simulation', default= 0)
    parser.add_argument('--NumberSims', type = int, help='Number of simulations done for convergence calculation', default  = 1000)
    parser.add_argument('--NumSweeps', type = int, help='Number of sweeps done per configuration', default  = 1500)
    parser.add_argument('--SaveData', type = str, help='Name of file saving data to', default  = None)
    args = parser.parse_args()

    if args.Data != 0:
        if args.Data == 1:  # Convergence steps for Game of Life
            print("Running convergence analysis")
            DataObj = DataAcquisition(args.NumberSims, GameOfLifeLattice, args.Width, args.Length)
            DataObj.converge_data()
            if args.SaveData:
                DataObj.save_data(args.SaveData)

        elif args.Data == 2:  # Glider simulation
            print("Running glider analysis")
            DataObj = DataAcquisition(args.NumberSims, GameOfLifeLattice, args.Width, args.Length)
            DataObj.GliderVel(args.NumSweeps)
            if args.SaveData:
                DataObj.save_data(args.SaveData)

    else:
        Lattice = GameOfLifeLattice(args.Length, args.Width, args.Seed)

        if args.Anim == False:
            for _ in range(args.NumSweeps):
                Lattice.RunNoAnim()
        else:
            anim = Lattice.animate_fig(args.NumSweeps)
            plt.show()