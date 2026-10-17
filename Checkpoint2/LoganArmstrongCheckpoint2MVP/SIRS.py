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


class SIRSLattice:
    def __init__(self, length, width, P_SI, P_IR, P_RS,P_Immunity, seed):
        self.length = length
        self.width = width
        self.p_si = P_SI
        self.p_ir = P_IR
        self.p_rs = P_RS
        self.immunity = P_Immunity
        self.seed = seed

        if self.seed ==0:
            self.lattice = np.random.choice([0,1],(self.length,self.width)) #Randomly generate a lattice of values 0 and 1
            #Let 0 be susceptible, 1 be infected, and -1 be recovered
        if self.seed == 1: #Need dynamic equilibrium
            self.p_si = 0.3
            self.p_ir = 0.4
            self.p_rs = 0.3
            self.lattice = np.random.choice([0,1],(self.length,self.width))
        if self.seed ==2: #Need absorbing state
            self.p_si = 0.15
            self.p_ir = 0.55
            self.p_rs = 0.3
            self.lattice = np.random.choice([0,1],(self.length,self.width))
        if self.seed == 3:
            self.lattice = np.random.choice([0,1],(self.length,self.width)) #Randomly generate a lattice of values 0 and 1
            #Set some points as -2 to be permanently recovered/immune
            random_indices = np.random.choice(self.lattice.size,size = int(self.lattice.size*self.immunity),replace = False)
            self.lattice.flat[random_indices]=-2

        self.infected_count = []
        self.suscept_count = []
        self.recovered_count = []
        self.infected_frac = []
        # self.infected_count.append(np.sum(self.lattice==1))
        # self.suscept_count.append(np.sum(self.lattice==0))
        # self.recovered_count.append(0) #No one has recovered initially
    def update_point(self):
        """Updates points in SIRS lattice according to 4 nearest neighbors status"""

        selected_x = random.randint(0, self.width - 1)
        selected_y = random.randint(0, self.length - 1)

        inst_prob = random.random() #There is probability for change in every circumstance, a value will be needed in each case therefore this is not a slowdown

        if self.lattice[selected_y, selected_x] == 0:  #Selecting a susceptible point
            if ((self.lattice[(selected_y-1)%self.length, selected_x]==1)|  #Checking site above
                (self.lattice[(selected_y+1)%self.length, selected_x]==1)|#Checking site below
                (self.lattice[selected_y,(selected_x-1)%self.width]==1)|#Checking site left
                (self.lattice[selected_y,(selected_x+1)%self.width]==1)):#Checking site right
                # (self.lattice[(selected_y-1)%self.length,(selected_x-1)%self.width]==1)|#Checking site northwest
                # (self.lattice[(selected_y-1)%self.length,(selected_x+1)%self.width]==1)|#Checking site northeast
                # (self.lattice[(selected_y+1)%self.length,(selected_x+1)%self.width]==1)|#Checking site southeast
                # (self.lattice[(selected_y+1)%self.length,(selected_x-1)%self.width]==1)):#Checking site southwest
                #Above condition checks if any of the nearest neighbors are infected
                if inst_prob<=self.p_si:
                    self.lattice[selected_y,selected_x] =1 #If the instantaneous probability is LEQ to inputted suscept to infect prob, site is now infected
        
        
        
        elif (((self.lattice[selected_y,selected_x]==1) & (inst_prob<= self.p_ir))|((self.lattice[selected_y,selected_x]==-1) & (inst_prob<= self.p_rs))):
            self.lattice[selected_y,selected_x]=((self.lattice[selected_y,selected_x]+1)/(-2*self.lattice[selected_y,selected_x]))
            #Case 1: selected point is infected i.e., =1 --> 1+1/(-2*1)=-1 works
            #Case 2: selected point is recovered i.e., =-1 --> 1+-1/(-2*-1) = 0/2 = 0 works
    def update_counts(self):
        """Updates the internal list of infected count, susceptable count, recovered count, and the fraction of infected"""
        self.infected_count.append(np.sum(self.lattice==1))
        self.suscept_count.append(np.sum(self.lattice==0))
        self.recovered_count.append(np.sum(self.lattice==-1))
        self.infected_frac.append(np.sum(self.lattice==1)/(self.length*self.width))

    def gen_fig(self):
        """Generates initial figures for animation"""
        self.fig, self.ax = plt.subplots(figsize = (8,6))
        self.im = self.ax.imshow(self.lattice, cmap='viridis', aspect='auto', vmin = -1, vmax = 1)
        #Need to manually set the minimum and maximum values on the lattice (vmin and vmax), as when we initialize the lattice it will only be on a range of 0-1
        #As such, when cells recover this will not be visualized and the animation will be useless
        self.ax.set_title('SIRS Infection Model')
        plt.colorbar(self.im, ax=self.ax)

    def update(self, frame):
        """Updates the frames for animation"""
        #self.livecount = 0
        for _ in range(self.length*self.width): #Does one full sweep
            self.update_point()
        if (frame>=500): #500 sweeps to equilibrate, update number of counts every 10 frames to avoid incredibly rapid oscillations
            self.update_counts()
        self.im.set_data(self.lattice)
        return self.im
    def animate_fig(self,frames):
        """Creates a functioning animation for the SIRS lattice"""
        self.gen_fig()
        anim =animation.FuncAnimation(self.fig, self.update, frames, repeat = False, interval = 40)
        return anim
    def gen_plots(self):
        """Generate plot comparing number of infected, susceptible and recovered counts
           Recall the first 500 frames are used as equilibration and are not recorded!
        """


        plt.figure(figsize=(6, 4))
        plt.plot(self.infected_count, color = 'yellow',label = 'Infected Counts')
        plt.plot(self.suscept_count, color = 'blue', label = 'Susceptible Counts')
        plt.plot(self.recovered_count, color = 'purple', label = 'Recovered Counts')
        plt.title("Number of counts")
        plt.xlabel(f'Frame')
        plt.ylabel("# Of Cells")
        plt.legend(loc = 'upper right')
        plt.show()

    def RunNoAnim(self):
        """Runs code without any animation, to be used for data collection"""
        for _ in range(self.length*self.width): #Does one full sweep
            self.update_point()
        self.update_counts() #Update counts only after a sweep


class DataAcquisition:
    def __init__(self, num_sims, LatticeClass, width, length):
        self.num_sims = num_sims
        self.width = width
        self.length = length
        self.im_list = []
        self.avg_infected = []
        self.infected_variance_error = []
        self.latticetype = LatticeClass

    def FracImmune(self,num_sweeps, P_immune,P_immune_final,resolution):
        """Finds the average number of infected for different immunity percentages"""
        R = resolution
        iv = (P_immune_final-P_immune)/R
        while(P_immune<=P_immune_final):
            inst_lattice = self.latticetype(50,50,0.5,0.5,0.5,P_immune,3)
            for _ in range(num_sweeps):
                inst_lattice.RunNoAnim()
            self.im_list.append(P_immune)
            self.avg_infected.append(s.mean(inst_lattice.infected_frac[100:]))#Skip the first 100 sweeps as these are equilibration
            print(f'Trial done')
            P_immune+=iv
        
        plt.figure(figsize=(8,6))
        plt.plot(self.im_list,self.avg_infected)
        plt.title(f"Infection Fraction vs Population Immunity")
        plt.xlabel(f'Fraction of Population With Immunity')
        plt.ylabel(f'Average Fraction of Population Infected After {num_sweeps} Sweeps')
        plt.show()

    def CalcError(self, numpy_list, num_sweeps):
        """Simple jacknife algorithm to calculate uncertainty/error from a given numpy list"""
        #Use jacknife method from ising model
        resampled_variances = []
        for i in range(num_sweeps):
            resample = np.random.choice(numpy_list, len(numpy_list), replace = True) #Resamples list
            resampled_variances.append((np.var(resample)))#Calculates variance of the resampled list
        return np.std(resampled_variances)

    def HeatmapSIRS(self,num_sweeps,resolution):
        """Generates a heatmap of the average value of infected fraction as both P_si and P_rs vary"""
        #Resolution controls the steps of increase for both P_si and P_rs
        R = resolution
        P_si = 0.05
        avg_value_list = np.zeros((R,R))
        iv = 1/R #Increment value
        tick_list = []

        for j in range(R):
            P_si = (j+1)*iv
            tick_list.append(f"{P_si:.3f}") #Formats as a float with 2 decimal points to avoid super long axes ticks (from floating point imprecision)
            for i in range(R):
                P_rs = (1+i)*iv
                print(f'Sanity Check')
                inst_lattice = self.latticetype(self.width,self.length,P_si,0.5, P_rs,0,0)
                for k in range(num_sweeps):  #Do 1500 sweeps
                    inst_lattice.RunNoAnim()
                avg_value_list[j,i]=s.mean(inst_lattice.infected_frac)
            print(f'Value calculated for P_si = {P_si}')
        ax =sns.heatmap(avg_value_list, cmap = 'coolwarm', square = True, cbar_kws={'label': 'Fraction of system infected'}, xticklabels=tick_list, yticklabels=tick_list)
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_yticks(ax.get_yticks()[::2])
        plt.xlabel('P_rs, probability of recovered cell to become susceptible')
        plt.ylabel('P_si, probability of susceptible cell to become infected')
        plt.gca().invert_yaxis() #Reverses the y axis, only necessary because of how calculations are done, purely cosmetic
        plt.tight_layout()
        plt.show()

    def InfectionVariance(self, num_sweeps, resolution):
        """Calculates the variance(and uncertainty) in the infected fraction as a function of P_si, returns these values plotted with error bars """
        R = resolution
        #P_rs = 0.5 = P_ir
        N=self.length * self.width
        infected_variance_list = []
        infected_variance_error_list = []
        P_si_list = []
        Infected_frac_test = []
        Infected_frac_Uncert =[]
        iv = 1/R
        for i in range(R+1):
            P_si = (i*iv)
            inst_lattice = self.latticetype(self.width, self.length, P_si,0.5,0.5,0,0)
            for _ in range(100+num_sweeps):  #Equilibrium sweeps(100)+Data collection (1000)
                inst_lattice.RunNoAnim()
            inst_infected_list = np.array(inst_lattice.infected_count[100:]) #Selects all values past 100th, remember 100 equilibrium sweeps 
            #print(inst_infected_list)
            inst_val = np.var(inst_infected_list)/N  #Calculates (<I^2>-<I>^2)/N
            infected_variance_list.append(inst_val)#Appends the variance 
            
            #raw_error_vals = self.CalcError(inst_infected_list,num_sweeps)

            infected_variance_error_list.append(self.CalcError(inst_infected_list,num_sweeps)/N)

            
            P_si_list.append(P_si)
            print(f'{i}th run done')
        self.infected_variance_error = infected_variance_error_list
        plt.figure(figsize=(8,6))
        plt.plot(P_si_list,infected_variance_list)
        plt.errorbar(P_si_list,y=infected_variance_list,yerr=infected_variance_error_list)
        plt.title(f"Variance in Average Infection Fraction vs P_SI")
        plt.xlabel(f'P_si, probability of going from susceptible to infected')
        plt.ylabel(f'Variance in Average Infection Fraction')
        #plt.gca().invert_xaxis() #Done in lecture, likely want here too
        plt.show()

    def SirsWave(self):
        """Creates a wave pattern in the SIRS lattice"""
        WaveLattice = self.latticetype(self.length,self.width,0.8,0.1,0.01,0,0)
        anim = WaveLattice.animate_fig(self.num_sims)
        plt.show()
        plots = WaveLattice.gen_plots()
        plt.show()

    def save_data(self, filename):
        """Save all collected data to a file."""
        with open(filename + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['im_list', 'avg_infected', 'error_avg_infected'])
            all_lists = [self.im_list, self.avg_infected, self.infected_variance_error]
            max_len = max(len(l) for l in all_lists)
            for i in range(max_len):
                writer.writerow([l[i] if i < len(l) else '' for l in all_lists])
        print(f"Data saved to {os.path.abspath(filename + '.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Width', type = int, help='Width of the created lattice', default  = 50)
    parser.add_argument('--Length', type = int, help='Length of the created lattice', default  = 50)
    parser.add_argument('--P_SI', type = float, help='Probability of going from susceptible to infected', default  = 0.5)
    parser.add_argument('--P_IR', type = float, help='Probability of going from infected to recovered', default  = 0.5)
    parser.add_argument('--P_RS', type = float, help='Probability of going from recovered to susceptible', default  = 0.5)
    parser.add_argument('--P_Immune', type = float, help='Probability of populace being immune', default  = 0)
    parser.add_argument('--Seed', type = int, help='Seed of the initial state, 0 is fully random, 1 is a dynamic equilibrium, 2 is an absorbing state' \
    '3 is immune state', default  = 0)
    parser.add_argument('--Anim', type = bool, help='If set to True animates the dynamics', default  = False)
    parser.add_argument('--Data', type = int, help='The desired data to be acquired and plotted' \
    '0 = None, 1 = Immunity fraction trials, 2 = Heatmap/Contour plot, 3 = Variance in infected counts' \
    '4 = SIRS wavepattern', default= 0)
    parser.add_argument('--NumberSims', type = int, help='Number of simulations done for data acquisition', default  = 1000)
    parser.add_argument('--NumSweeps', type = int, help='Number of sweeps done per configuration for variance of infected counts and heatmap', default  = 1500)
    parser.add_argument('--Res', type = int, help='Resolution for data trials', default  = 20)
    parser.add_argument('--EquibSweeps', type = int, help='Number of sweeps done until equilibrium is met', default  = 100)
    parser.add_argument('--Prob_iminit', type = float, help='Initial percentage of population immune for SIRS immunity trials', default  = 0)
    parser.add_argument('--Prob_imfinal', type = float, help='Final percentage of population immune for SIRS immunity trials', default  = 1)
    parser.add_argument('--SaveData', type = str, help='Name of file saving data to', default  = None)
    args = parser.parse_args()

    if args.Data != 0:
        if args.Data == 1:  # Immunity fraction trials
            print("Running immunity fraction analysis")
            DataObj = DataAcquisition(args.NumberSims, SIRSLattice, args.Width, args.Length)
            DataObj.FracImmune(args.NumSweeps, args.Prob_iminit, args.Prob_imfinal,args.Res)
            if args.SaveData:
                DataObj.save_data(args.SaveData)
        
        elif args.Data == 2:  # Heatmap/Contour plot
            print("Running heatmap analysis")
            DataObj = DataAcquisition(args.NumberSims, SIRSLattice, args.Width, args.Length)
            DataObj.HeatmapSIRS(args.NumSweeps, args.Res)
            if args.SaveData:
                DataObj.save_data(args.SaveData)
        
        elif args.Data == 3:  # Variance in infected counts
            print("Running infection variance analysis")
            DataObj = DataAcquisition(args.NumberSims, SIRSLattice, args.Width, args.Length)
            DataObj.InfectionVariance(args.NumSweeps, args.Res)
            if args.SaveData:
                DataObj.save_data(args.SaveData)

        elif args.Data == 4:  # Wave Simulation
            print("Running Wave Simulation")
            DataObj = DataAcquisition(args.NumberSims, SIRSLattice, args.Width, args.Length)
            DataObj.SirsWave()
            #No need to save data here

    else:
        Lattice = SIRSLattice(args.Length, args.Width, args.P_SI, args.P_IR, args.P_RS, args.P_Immune, args.Seed)

        if args.Anim == False:
            for _ in range(args.NumSweeps):
                Lattice.RunNoAnim()
        else:
            anim = Lattice.animate_fig(args.NumSweeps)
            plt.show()