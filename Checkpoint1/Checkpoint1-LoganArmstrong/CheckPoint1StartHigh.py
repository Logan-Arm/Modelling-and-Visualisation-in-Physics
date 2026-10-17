import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import statistics as s
import argparse


class IsingLattice:
    """
    Represents an Ising model lattice with spin configurations and energy calculations.
    """
    
    def __init__(self, length, width, J, k, T):#Come back to this and change things around for kb and T if needed
        """
        Initialize the Ising lattice.
        

        """
        self.length = length
        self.width = width
        self.J = 1.0
        self.k = k 
        self.T = T #Initially high temperature chaotic
        self.lattice = np.random.choice([-1, 1], size=(length, width))
        self.Evalue = 0
        #self.total_mag = []
        #self.total_energy = []
        self.suscept_list = []
        self.Cv_list = []
        self.suscept_uncert = []
        self.Cv_uncert = []
        self.T_list = []
        self.Elist = []
        self.absmaglist = []
    


    def get_nearest_neighbor_energy(self, x_pos, y_pos):
        """
        Calculate the nearest neighbor energy contribution from a selected point.
        

        """
        point_val = self.lattice[y_pos, x_pos]
        
        # Use periodic boundary conditions
        y_above = (y_pos - 1) % self.length
        y_below = (y_pos + 1) % self.length
        x_left = (x_pos - 1) % self.width
        x_right = (x_pos + 1) % self.width
        
        # Sum of interactions with neighbors
        sumterm = (point_val * self.lattice[y_above, x_pos] + 
                   point_val * self.lattice[y_below, x_pos] +
                   point_val * self.lattice[y_pos, x_left] +
                   point_val * self.lattice[y_pos, x_right])
        
        energy = -self.J * sumterm
        return energy
    
    def get_total_energy(self):
        """
        Calculate the total energy of the lattice.
        Uses checkerboard pattern to avoid double counting.
        

        """
        energy = 0
        for i in range(self.length):
            for j in range(self.width):
                if (i + j) % 2 == 0:
                    energy += self.get_nearest_neighbor_energy(j, i)
        
        self.Evalue = energy
    

    # def starting_vals (self):
    #     self.E = self.get_total_energy

    def flip_spin(self, x_pos, y_pos):
        """Flip the spin at the specified position."""
        self.lattice[y_pos, x_pos] *= -1
    
    def swap_spins(self, x1, y1, x2, y2):
        """Swap spins at two positions."""
        self.lattice[y1, x1], self.lattice[y2, x2] = \
            self.lattice[y2, x2], self.lattice[y1, x1]
    
    def copy(self):
        """Create a deep copy of the lattice."""
        new_lattice = IsingLattice(self.length, self.width, self.J, self.k, self.T)
        new_lattice.lattice = np.copy(self.lattice)
        return new_lattice
    def get_total_mag(self):
        """Calculates the total magnetisation of the lattice"""
        return np.sum(self.lattice)#Trying without absolute value

    def calculate_uncert(self, parameter_list):
        """
        Calculates the uncertainty of a parameter when provided the list of parameter values
        """
        mean = np.mean(parameter_list)
        sq_of_mean = np.mean((parameter_list**2))
        uncert= math.sqrt(sq_of_mean-mean**2)
        return uncert    
    def calc_Cv(self, value_list):
        """Calculates the Cv when provided with a list of energy values"""
        inst_val = self.calculate_uncert(value_list)**2
        Cv = 1/(self.width*self.length*self.k*self.T**2)* inst_val
        return Cv
    def calc_Suscept(self, value_list):
        """Calculates the susceptibility of the lattice when provided with a list of magnetisation values"""
        inst_val = self.calculate_uncert(value_list)**2
        Suscept = 1/(self.width*self.length*self.k*self.T)* inst_val
        return Suscept
    
    def datarun(self,dynamics,n_counts,ntotal,equib_count):
        """
        Runs the desired dynamics without the use of an animation, meant for speedy data collection.  
        n_counts = number of counts per temperature
        equib_count = number of equilibrium counts per temperature
        """
        
        totcount = 0 #Total track of frames
        inst_mag = []
        inst_abs_mag = []
        inst_E = []
        fcount = 0 #Need to reset fcount each time to allow for equilibration
        
        #while totcount<=ntotal:
        #Above line is also a valid way to run the simulation, but requires you to make active calculations for the number of sweeps


        while self.T>=0.98: #Needs to be 0.98 due to floating point imprecision   
        #If want a different lower temperature bound alter here 
        
            for _ in range(self.length*self.width):
                dynamics.step(self)
            if fcount>=equib_count and fcount%10==0:
                inst_mag.append(self.get_total_mag())
                inst_abs_mag.append(abs(self.get_total_mag()))  #Stores the instant absolute value of the magnetisation 
                inst_E.append(self.Evalue)
            if fcount%(equib_count + n_counts)==0 and fcount !=0:
                self.Elist.append(s.mean(inst_E)) #Stores mean energy value
                self.absmaglist.append(s.mean(inst_abs_mag)) #Calculates and stores the average absolute value of magnetisation for this temperature
                E_cv_resample = []
                #Suscept_resample = []
                self.T_list.append(self.T)
                self.T-=0.1
                for i in range(int(n_counts/10)):  #Changing to be n_counts/10 
                    E_cv_resample.append(self.calc_Cv(np.random.choice(inst_E, len(inst_E)))) #These give small errors!! Making inst_E, len(inst_E)
                    #Suscept_resample.append(self.calc_Suscept(np.random.choice(inst_mag, len(inst_mag))))    
                #self.suscept_uncert.append(self.calculate_uncert(np.array(Suscept_resample)))
                ##Above lines can be uncommented to add uncertainty to the susceptibility
                self.Cv_uncert.append(self.calculate_uncert(np.array(E_cv_resample)))
                self.Cv_list.append(self.calc_Cv(np.array(inst_E)))
                self.suscept_list.append(self.calc_Suscept(np.array(inst_mag)))
                print(f"Magnetic susceptibility list is {self.suscept_list}")
                inst_mag = []
                inst_E = []
                inst_abs_mag = []
                self.get_total_energy()  #Added to hopefully energy divergence
                fcount =0 
            
                print(f"Data collected for temperature = {self.T+0.1}")
            if totcount%5000 ==0:
                print(totcount)
            fcount+=1
            totcount+=1
    def save_data_to_file(self, filename='ising_data.txt'):
        """
        Saves the internal data(susceptibility, Cv, uncertainty of Cv, average energy, and average absolute magnetisaion) of the lattice to a file labeled ising_data
        """
        with open(filename, 'w') as f:
            # Write header
            f.write("# Ising Model Simulation Data\n")
            f.write(f"# Lattice size: {self.length} x {self.width}\n")
            f.write(f"# J = {self.J}, k = {self.k}\n")
            f.write("#\n")
            f.write("# Columns: Temperature, Susceptibility, Heat_Capacity, Cv_Uncertainty, Average Energy, Average Absolute Magnetisation\n")
            f.write("#" + "-"*80 + "\n")
            
            # Write data
            for i in range(len(self.T_list)):
                T = self.T_list[i]
                suscept = self.suscept_list[i]
                #suscept_err = self.suscept_uncert[i]
                Cv = self.Cv_list[i]
                Cv_err = self.Cv_uncert[i]
                AverageEnergy = self.Elist[i]
                AverageAbsMag = self.absmaglist[i]
                
                f.write(f"{T:.6f}\t{suscept:.10f}\t{Cv:.10f}\t{Cv_err:.10f}\t{AverageEnergy:.10f}\t{AverageAbsMag:.10f}\n")
        
        print(f"Data saved to {filename}")



    def plotrelations(self):
        """
        Generates plotes of susceptibility,heat capacity, energy and mean magnetisation against Temperature
        """

        plt.figure(figsize=(6, 4))
        plt.plot(self.T_list, self.suscept_list)
        #plt.errorbar(self.T_list,self.suscept_list,yerr = self.suscept_uncert)
        plt.title("MagSusceptibility")
        plt.xlabel(f'Temperature')
        plt.ylabel("Magnetic Susceptibility")
        plt.show()


        plt.figure(figsize=(8, 6))
        plt.plot(self.T_list, self.Cv_list)
        plt.errorbar(self.T_list,self.Cv_list,yerr = self.Cv_uncert)
        plt.xlabel(f'Temperature')
        plt.ylabel("Heat Capacity")
        plt.title("Cv")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(self.T_list, self.Elist)
        plt.title("Mean Energy Vs Temp")
        plt.xlabel(f'Temperature')
        plt.ylabel("Average Energy")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(self.T_list, self.absmaglist)
        plt.title("Mean Absolute Magnetisation Vs Temp")
        plt.xlabel(f'Temperature')
        plt.ylabel("Average Absolute Magnetisation")
        plt.show()

    
    



class GlauberDynamics():
    """
    Glauber dynamics: single spin flip dynamics.
    """
    
    def step(self, lattice):
        """
        Perform one Glauber dynamics step (single spin flip).
        
        """
        # Select random position
        selected_x = random.randint(0, lattice.width - 1)
        selected_y = random.randint(0, lattice.length - 1)
        
        
        delta_E = -2*lattice.get_nearest_neighbor_energy(selected_x,selected_y)

        prob_flip = math.exp(-delta_E/(lattice.k*lattice.T))
        temp_prob = random.random()
        if delta_E<=0:
            lattice.Evalue+=delta_E
            lattice.flip_spin(selected_x, selected_y)
        elif temp_prob<prob_flip:
            lattice.Evalue+=delta_E
            lattice.flip_spin(selected_x, selected_y)
        else:
            pass

class KawasakiDynamics():
    """
    Kawasaki dynamics: spin exchange dynamics (conserves magnetization).
    """
    
    def step(self, lattice):
        """
        Perform one Kawasaki dynamics step (spin exchange).
        
        """
        # Select two different random positions
        x1 = random.randint(0, lattice.width - 1)
        y1 = random.randint(0, lattice.length - 1)
        
        x2 = random.randint(0, lattice.width - 1)
        while x2 == x1:
            x2 = random.randint(0, lattice.width - 1)
        
        y2 = random.randint(0, lattice.length - 1)
        while y2 == y1:
            y2 = random.randint(0, lattice.length - 1)

        #Doing only nearest neighbor change
        E_init = lattice.get_nearest_neighbor_energy(x1,y1) + lattice.get_nearest_neighbor_energy(x2,y2)
        lattice.swap_spins(x1,y1,x2,y2)
        E_final = lattice.get_nearest_neighbor_energy(x1,y1) + lattice.get_nearest_neighbor_energy(x2,y2) #Since swapped lattice in general don't need to swap indices


        
        delta_E = E_final - E_init #This should account for nearest neighbor terms
       # print(f"The change in energy is {delta_E}")
       ##Above line is sanity check
        prob_flip = math.exp(-delta_E/(lattice.k*lattice.T))
        temp_prob = random.random()
        if delta_E<=0:
            lattice.Evalue+=delta_E
        elif temp_prob<prob_flip:
            lattice.Evalue+=delta_E
        else:
            lattice.swap_spins(x1, y1, x2, y2) #Swaps lattice back if the flip is rejected



class ImShow:
    """
    Class function responsible for all animations within the code
    """
    def __init__(self, lattice, dynamics_model, n_frames, equib_count, n_readings):
        """Initializes all values necessary for the animation.
        n_frames = number of total frames
        n_readings = number of readings/sweeps per temperature
        equib_count = number of equilibration sweeps per temperature
        """
        self.lattice = lattice
        self.dynamics_model = dynamics_model
        self.n_frames = n_frames
        self.equib_count = equib_count
        self.nreadings = n_readings
        self.N = self.lattice.width*self.lattice.length


    def createplot(self):
        """
        Initializes the plot to be used for animation
        """
        self.fig, self.ax = plt.subplots(figsize = (8,6))
        self.im = self.ax.imshow(self.lattice.lattice, cmap='viridis', aspect='auto')
        self.ax.set_title('Ising Lattice')
        plt.colorbar(self.im, ax=self.ax)
        self.inst_E = []
        self.inst_mag = []
    
    def update(self, frame):
        """
        Runs the desired dynamic model, calculates data, and updates the animation
        """
        #print(f"On frame {frame}")
        for _ in range(self.N):
            self.dynamics_model.step(self.lattice)
        if frame > self.equib_count and frame%10 ==0: #Animate every 10 frames beyond equilibration
            self.im.set_data(self.lattice.lattice)
            self.inst_mag.append(self.lattice.get_total_mag())
            self.inst_E.append(self.lattice.Evalue)
        if frame %(self.nreadings+self.equib_count)==0 and frame != 0:
            E_cv_resample = []
            Suscept_resample = []
            self.lattice.T_list.append(self.lattice.T)
            self.lattice.T-=0.1
            for i in range(self.nreadings):
                E_cv_resample.append(self.lattice.calc_Cv(np.random.choice(self.inst_E, self.N)))
                Suscept_resample.append(self.lattice.calc_Suscept(np.random.choice(self.inst_mag, self.N)))
            
            self.lattice.suscept_uncert.append(self.lattice.calculate_uncert(np.array(Suscept_resample)))
            self.lattice.Cv_uncert.append(self.lattice.calculate_uncert(np.array(E_cv_resample)))
            self.lattice.Cv_list.append(self.lattice.calc_Cv(np.array(self.inst_E)))
            self.lattice.suscept_list.append(self.lattice.calc_Suscept(np.array(self.inst_mag)))
            print(f"Magnetic susceptibility list is {self.lattice.suscept_list}")
            self.inst_mag = []
            self.inst_E = []
            print(f"Data collected for temperature = {self.lattice.T+0.1}")
        return [self.im]
    def animate(self, interval=10):
        """Create and return the animation"""
        self.createplot()
        anim = animation.FuncAnimation(
            self.fig, 
            self.update, 
            frames=self.n_frames,
            interval=interval,
            blit=True,
            repeat=False
        )
        return anim




        


if __name__ == "__main__":
    
    #Adding parsing section:
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type = int,default  = 50)
    parser.add_argument('--width', type = int, default = 50)
    parser.add_argument('--J', type = int, default = 1)
    parser.add_argument('--k', type = int, default = 1)
    parser.add_argument('--T', type = float, default = 3)
    parser.add_argument('--Dynamics', type = int, help='Glauber = 0, Kawasaki = 1', default  = 0)
    parser.add_argument('--SweepsPerTemp', type = int, help = 'How many sweeps per temperature (after equilibrium)', default = 10000)    
    parser.add_argument('--EquibSweep', type = int, help = 'How many sweeps till equilibrium', default = 1000)
    parser.add_argument('--TotalSweeps', type = int, help = 'How many total sweeps', default = 220000)
    parser.add_argument('--Animate', type = int, help = 'Animation off by default, set 1 to create animation', default = 0)
    args = parser.parse_args()

    lattice = IsingLattice(args.length, args.width, args.J, args.k, args.T)
    lattice.get_total_energy()
    if args.Dynamics == 0:
        dynamics = GlauberDynamics()
    else:
        dynamics = KawasakiDynamics()

    if args.Animate == 1:
        visualizer = ImShow(
        lattice=lattice,
        dynamics_model=dynamics,
        n_frames=args.TotalSweeps,
        equib_count=args.EquibSweep,
        n_readings=args.SweepsPerTemp)
        # Animate
        anim = visualizer.animate(interval=5)
        plt.show()
    else:
        lattice.datarun(dynamics,args.SweepsPerTemp,args.TotalSweeps,args.EquibSweep)
    lattice.plotrelations()
    lattice.save_data_to_file()
    
    
    

