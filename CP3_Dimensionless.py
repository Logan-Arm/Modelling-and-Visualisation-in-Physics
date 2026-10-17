import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import random
import statistics as s
import seaborn as sns
import argparse
import csv
from numba import njit, prange


"""Using numba to create a lattice potential function which will be called in my Cahn hiliard class"""
@njit(parallel = True)
def calc_potential(lattice, dx, l, w):
    # Dimensionless form: mu = -phi + phi^3 - nabla^2(phi)
    # All parameters (a, b, k, M) absorbed into dimensionless units
    pot = np.zeros_like(lattice) #Creates a lattice of zeros in the same size as whatever is inputed
    for i in prange(l):  #THIS PARALLELISES ACROSS ROWS!!!
        for j in range(w):
            laplac = (lattice[(i+1)%l,j] #Gets point up accounting for boundary
                      +lattice[(i-1)%l,j]#point down
                      +lattice[i,(j+1)%w]#point right
                      +lattice[i,(j-1)%w] #point down
                      -4*lattice[i,j])/(dx**2)
            pot[i,j] = -lattice[i,j] + lattice[i,j]**3 - laplac
    return pot


"""Using numba to create an update function for future lattice, meant for Cahn Hiliard"""
@njit(parallel=True)
def update_lattice(lattice, pot, dt, dx, l ,w):
    # Dimensionless so --> dphi/dt = nabla^2(mu), i.e., M absorbed into time units
    lattice_fut = np.zeros_like(lattice)#Creates a lattice of zeros in the same size as whatever is inputed
    for i in prange(l): #This parallelizes
        for j in range(w):
            pot_term = (pot[(i+1)%l,j]
                        +pot[(i-1)%l,j]
                        +pot[i,(j+1)%w]
                        +pot[i,(j-1)%w]
                        -4*pot[i,j])/(dx**2)
            #Above calculates the contribution to the updated lattice from the potential lattice
            lattice_fut[i,j] = lattice[i,j] + dt*pot_term
    return lattice_fut

"""Using Numba to calculate gradient squared"""
@njit(parallel=True)
def calc_grad_sq(lattice, dx, l, w):
    grad_sq = np.zeros_like(lattice)
    for i in prange(l):
        for j in range(w):
            grad_x = (lattice[(i+1)%l,j]-lattice[(i-1)%l,j])/(2*dx)
            grad_y = (lattice[i,(j+1)%w]-lattice[i,(j-1)%w])/(2*dx)
            grad_sq[i,j] = grad_x**2 + grad_y**2
    return grad_sq

"""Using Numba to calculate gradient in 2D"""
@njit(parallel = True)
def calc_grad2D(lattice,dx,l,w):
    grad_x_array = np.zeros_like(lattice)
    grad_y_array = np.zeros_like(lattice)
    for i in prange(l):
        for j in range(w):
            grad_x = (lattice[(i+1)%l,j]-lattice[(i-1)%l,j])/(2*dx)
            grad_y = (lattice[i,(j+1)%w]-lattice[i,(j-1)%w])/(2*dx)
            grad_x_array[i,j] = grad_x
            grad_y_array[i,j] = grad_y
    return grad_x_array, grad_y_array

"""Using Numba to calculate gradient in 3D"""
@njit(parallel = True)
def calc_grad3D(lattice,dx,l,w,h):
    grad_x_array = np.zeros_like(lattice)
    grad_y_array = np.zeros_like(lattice)
    grad_z_array = np.zeros_like(lattice)
    for i in prange(l):
        for j in range(w):
            for k in range(h):
                grad_x = (lattice[(i+1)%l,j,k]-lattice[(i-1)%l,j,k])/(2*dx)
                grad_y = (lattice[i,(j+1)%w,k]-lattice[i,(j-1)%w,k])/(2*dx)
                grad_z = (lattice[i,j,(k+1)%h]-lattice[i,j,(k-1)%h])/(2*dx)
                grad_x_array[i,j,k] = grad_x
                grad_y_array[i,j,k] = grad_y
                grad_z_array[i,j,k] = grad_z
    return grad_x_array, grad_y_array, grad_z_array



"""Using Numba for free energy calculation"""
@njit(parallel = True)
def free_energy(lattice, grad_sq, l, w):
    # Dimensionless form: f = -phi^2/2 + phi^4/4 + (nabla phi)^2 / 2
    # Corresponds to a=b=k=1 in dimensional form which is ran in the DimensionFe_fix file
    f = np.zeros_like(lattice)
    for i in prange(l):
        for j in range(w):
            f[i,j] = (-0.5*lattice[i,j]**2
                      + 0.25*lattice[i,j]**4
                      + 0.5*grad_sq[i,j]
            )
    return f

"""Using Numba for the potential calculation in Poisson using jacobian methods"""
@njit(parallel = True)
def update_poisson_3d(lattice,density_lattice,l,w,h):
    new_lat = np.zeros_like(lattice)
    for i in prange(1,l-1): #Needs the 1,x-1 to ignore the boundary terms which are always zero
        for j in range(1,w-1):
            for k in range(1,h-1):
                new_lat[i,j,k] = (1/6)*(lattice[(i+1)%l,j,k]+lattice[(i-1)%l,j,k]+ 
                                  lattice[i,(j+1)%w,k]+
                                  lattice[i,(j-1)%w,k]+
                                  lattice[i,j,(k+1)%h]+
                                  lattice[i,j,(k-1)%h]+
                                  density_lattice[i,j,k]
                                  )
    return new_lat

"""Using Numba for creating a random array of charges"""
@njit(parallel = True)
def random_charge_dist(lattice,l,w,h,charge_frac):
    charge_dist = np.zeros_like(lattice)
    for i in prange(1,l-1):
        for j in range(1,w-1):
            for k in range(1,h-1):
                inst_chance = np.random.rand()
                if inst_chance <= charge_frac:
                    charge_dist[i,j,k] = 1
                else:
                    charge_dist[i,j,k] = 0
    return charge_dist

"""Using Numba to calculate the magnetic field for Poisson wire through center"""
@njit(parallel = True)
def mag_cross(lattice,l,w,dx):
    Bx = np.zeros((l,w,),dtype = np.float64)
    By = np.zeros((l,w,),dtype = np.float64)
    for i in prange(1,l-1): #Ignore boundaries (boundaries effectively infinitely far away)
        for j in range(1,w-1):
            Bx[i,j] = (lattice[i,j+1]-lattice[i,j-1])/(2*dx) 
            By[i,j] = (lattice[i+1,j]-lattice[i-1,j])/(2*dx)
    return Bx,By

"""Using Numba to create an update rule for the magnetic Poisson case using jacobian methods"""
@njit(parallel = True)
def update_poisson_mag(lattice, density_lattice, l, w):
    new_lat = np.zeros_like(lattice)
    for i in prange(1,l-1): #Needs the 1,x-1 to ignore the boundary terms which are always zero
        for j in range(1,w-1):
                new_lat[i,j] = (1/4)*(lattice[(i+1)%l,j]+lattice[(i-1)%l,j]+ 
                                  lattice[i,(j+1)%w]+
                                  lattice[i,(j-1)%w]+
                                  density_lattice[i,j]
                                  )
    return new_lat

                
"""Using Numba for Gauss-Seidel 3d integration"""
@njit
def gauss_seidel_3d(lattice, density_lattice, l, w, h):
    for i in range(1, l-1):
        for j in range(1, w-1):
            for k in range(1, h-1):
                lattice[i,j,k] = (1/6)*(lattice[i+1,j,k] + lattice[i-1,j,k] +
                                         lattice[i,j+1,k] + lattice[i,j-1,k] +
                                         lattice[i,j,k+1] + lattice[i,j,k-1] +
                                         density_lattice[i,j,k])
    return lattice

"""Using Numba for Gauss-Seidel 2d integration"""
@njit
def gauss_seidel_2d(lattice, density_lattice, l, w):
    for i in range(1, l-1):
        for j in range(1, w-1):
                lattice[i,j] = (1/4)*(lattice[i+1,j] + lattice[i-1,j] +
                                         lattice[i,j+1] + lattice[i,j-1] +
                                         density_lattice[i,j])
    return lattice


"""Using Numba for Gauss-Seidel 3d SOR integration"""
@njit
def gauss_seidel_SOR_3d(lattice, density_lattice, l, w, h,sor_w):
    for i in range(1, l-1):
        for j in range(1, w-1):
            for k in range(1, h-1):
                lattice[i,j,k] = (1-sor_w)*lattice[i,j,k]+sor_w*((1/6)*(lattice[i+1,j,k] + lattice[i-1,j,k] +
                                         lattice[i,j+1,k] + lattice[i,j-1,k] +
                                         lattice[i,j,k+1] + lattice[i,j,k-1] +
                                         density_lattice[i,j,k]))
    return lattice


"""Using Numba for Gauss-Seidel 2d SOR integration"""
@njit
def gauss_seidel_SOR_2d(lattice, density_lattice, l, w,sor_w):
    for i in range(1, l-1):
        for j in range(1, w-1):
                lattice[i,j] = (1-sor_w)*lattice[i,j]+sor_w*(1/4)*(lattice[i+1,j] + lattice[i-1,j] +
                                         lattice[i,j+1] + lattice[i,j-1] +
                                         density_lattice[i,j])
    return lattice


"""Using Numba to create a 'distance-lattice' essentially describing each points distance from the origin, to be used for charge in center of box and wire calculations"""
@njit(parallel = True)
def dist_to_cent(lattice,l,w):
    dist_lat = np.zeros_like(lattice)
    for i in prange(1,l-1): #Borders are at infinity! do not include their distances
        for j in range(1,w-1):
            dist_lat[i,j] = np.sqrt((i-(l//2))**2+(j-(w//2))**2) #appends sqrt((i-center)**2 + (j-center)**2)
    return dist_lat




class Cahn_Hilliard:
    # All parameters are dimensionless:
    #   length scale: xi = sqrt(k/a)
    #   time scale:   tau = k / (M * a^2)
    # The equation in dimensionless form has no free parameters,
    # so a, b, k, M are no longer arguments.
    def __init__(self, length, width, deltax, deltat, phi0):
        """Initializes the Cahn-Hilliard lattice where all relavent constants i.e., a, k, b , and m have been absorbed into dx and dt.  
        dx is in units of sqrt(k/a), and dt is in units of k/(ma^2)
        """
        self.l = length
        self.w = width
        self.dx = deltax   # in units of xi = sqrt(k/a)
        self.dt = deltat   # in units of tau = k/(M*a^2)
        self.phi0 = phi0
        self.lattice = np.full((self.l,self.w), self.phi0, dtype = np.float64)#Initializes a full lattice of dimensions length x width with values phi0 in each spot
        self.lattice += np.random.uniform(-0.1, 0.1, (self.l, self.w))  # Creates background noise to allow for fluctuations

        self.pot_lattice = np.zeros_like(self.lattice) #Creates zero lattice of size and shape of self.lattice

        self.warmup()

    def warmup(self):
        """This simply runs all the numba functions which the Cahn-Hilliard class uses.  Numba works on a lower level than python traditionally does, and therefore 
        requires activation or compilation prior to being run.  This is done using a very small sized lattice for speeds sake."""
        dummy = np.zeros((4, 4), dtype=np.float64)
        dummy_pot = np.zeros_like(dummy)
        calc_potential(dummy, self.dx, 4, 4)
        update_lattice(dummy, dummy_pot, self.dt, self.dx, 4, 4)
        calc_grad_sq(dummy, self.dx, 4, 4)

    def potential_lattice(self):
        """Calculates the potential from the lattice phi value configuration using the numba potential function"""
        self.pot_lattice = calc_potential(
            self.lattice, self.dx, self.l, self.w
        )

    def gradient_sq(self):
        """Simply returns the gradient squared of the lattice"""
        return calc_grad_sq(self.lattice, self.dx, self.l, self.w)

    def free_energy_lattice(self):
        """Calculates the free energy of the lattice as specified in the checkpoint handout using a numba function."""
        grad_sq = self.gradient_sq()
        f = free_energy(
            self.lattice, grad_sq, self.l, self.w
        )
        # self.free_E = (np.sum(f) * self.dx**2)/(self.l *self.w)  #This makes it so its free energy per lattice point, a much easier number to deal with  
        self.free_E = np.sum(f)
    def free_energy_data(self,equilib_condit, tolerance, max_frame): #TOLERANCE IS NEEDED OTHERWISE CONDITION WILL NEVER BE MET DUE TO FLOAT IMPRECISION
        """Runs/updates the Cahn-Hilliard lattice for a number of frames specified by the max_frame parameter, recording the lattice free energy at each point.
        This is done to capture the curve/behavior of the lattice's free energy for different phi values.
        """
        equib_counter = 0
        self.free_energy_list = []

        frame_count = 0
        self.free_energy_lattice()
        while (frame_count <max_frame):
            self.potential_lattice()
            self.lattice = update_lattice(self.lattice, self.pot_lattice, self.dt, self.dx, self.l, self.w)
            self.free_energy_lattice()
            self.free_energy_list.append(self.free_E)

            frame_count+=1
            if frame_count%100 ==0:
                print(f'At frame {frame_count}')
        plt.figure(figsize=(10,6))
        plt.plot(self.free_energy_list)
        plt.xlabel(f'Time (dimensionless units $\\tau$)')
        plt.ylabel(f'Free Energy (dimensionless)')
        plt.title(f'Free Energy vs Time for phi_0 = {self.phi0}')
        plt.show()
        
    def gen_fig(self):
        self.fig, self.ax = plt.subplots(figsize = (8,6))
        self.im = self.ax.imshow(self.lattice, cmap='coolwarm', aspect='auto')
        self.ax.set_title('Cahn Hilliard Lattice')
        plt.colorbar(self.im, ax=self.ax)

    def update(self, frame):
        self.potential_lattice() #Updates the potential from new points
        self.lattice = update_lattice(self.lattice, self.pot_lattice, self.dt, self.dx, self.l, self.w)
        self.im.set_data(self.lattice)
        return self.im

    def animate_fig(self):
        """Generates an animation of the Cahn-Hilliard lattice"""
        self.gen_fig()
        anim =animation.FuncAnimation(self.fig, self.update,100000 , repeat = False, interval = 1)#1000 frames as a placeholder
        return anim

class Poisson:
    def __init__(self,length,width,height,seed,frac_charges,dx,permittivity,threshold,converg_criteria,SOR_w):
        """Initializes the poisson lattice, the only constant which changes is the permittivity
        Seed corresponds to the specific lattice we want to make, 0 corresponds to a single charge randomly placed in the box, 1 to a single charge 
        in the center of the box, 2 are charges placed randomly according to the fraction of charges (frac_charges), and 3 is a wire running along
        the z axis.

        Threshhold is the relative change which defines a convergence event
        Convergence criteria is how many convergence events is necessary before simulation is terminated
        SOR_w is the omega value used for SOR update algorithms
        """

        self.l = length
        self.w = width
        self.h = height
        self.seed = seed
        self.frac_charges = frac_charges
        self.dx = dx
        self.perm = permittivity
        self.thresh = threshold
        self.criteria = converg_criteria
        self.sorw = SOR_w
        self.convergence_iter=0
        self.mu0 = (4*np.pi)*10**(-7) #Mu_0 = 4pi * 10^-7
        
        if self.seed==0: #Single charge in the box
            self.lattice = np.zeros((self.l,self.w,self.h), dtype = np.float64)  #Creates array of zeros
            charge_i = random.randint(1,self.l-1)
            charge_j = random.randint(1,self.w-1)
            charge_k = random.randint(1,self.h-1)
            self.lattice[(charge_i,charge_j,charge_k)]=(1*(dx**2)/self.perm) #Making dimensionless (this effectively changes nothing but to account for any fluctuations good to include)
            self.charge_density = np.copy(self.lattice)#Creates a copy of the initial lattice to be used as a charge density array later

        if self.seed==1: #Single charge in center of the box
            self.lattice = np.zeros((self.l,self.w,self.h), dtype = np.float64)  #Creates array of zeros
            self.lattice[(self.l//2,self.w//2,self.h//2)]=(1*(dx**2)/self.perm) #Making dimensionless (this effectively changes nothing but to account for any fluctuations good to include)
            self.charge_density = np.copy(self.lattice)#Creates a copy of the initial lattice to be used as a charge density array later
        
        if self.seed ==2:#Randomly placed charges according to the fractional charge value
            self.lattice = np.zeros((self.l,self.w,self.h), dtype = np.float64)
            self.charge_density = self.charge_dist()*(self.dx**2)/(self.perm)
            self.lattice +=self.charge_density

        if self.seed ==3: #Wire through origin running along z-axis
            self.lattice = np.zeros((self.l,self.w,self.h), dtype = np.float64)
            self.lattice[self.l//2,self.w//2,:]=(1*(dx**2)/self.perm) #Making dimensionless (this effectively changes nothing but to account for any fluctuations good to include)
            self.charge_density = np.copy(self.lattice)

        if self.seed==4: #Used for magnetic test with a wire running through the center along z --> effectively a 2D system
            self.lattice = np.zeros((self.l,self.w), dtype = np.float64)
            self.lattice[self.l//2,self.w//2] = 1*(dx**2) #Charge in the center, should be scaled by mu_nought potentially
            self.charge_density = np.copy(self.lattice)
        self.warmup()
    
    def warmup(self): #Initializes all the numba functions which will be used
        """This simply runs all the numba functions which the Cahn-Hilliard class uses.  Numba works on a lower level than python traditionally does, and therefore 
        requires activation or compilation prior to being run.  This is done using a very small sized lattice for speeds sake."""
        dummy = np.zeros((4, 4, 4), dtype=np.float64)
        dummy2d = np.zeros((4, 4), dtype=np.float64)
        #All 3d functions
        calc_grad3D(dummy, 1, 4, 4, 4)
        update_poisson_3d(dummy, dummy, 4, 4, 4)
        random_charge_dist(dummy, 4, 4, 4, 0.1)
        gauss_seidel_3d(dummy,dummy,4,4,4)
        gauss_seidel_SOR_3d(dummy,dummy,4,4,4,1.6)
        
        #Now all 2d
        mag_cross(dummy2d,4,4,.1)
        update_poisson_mag(dummy2d,dummy2d,4,4)
        gauss_seidel_2d(dummy2d,dummy2d,4,4)
        gauss_seidel_SOR_2d(dummy2d,dummy2d,4,4,1.2)
        dist_to_cent(dummy2d,2,2)

    def charge_dist(self):
        """Randomly assigns lattice points to be charges depending on the fractional charge described"""
        charge_dist = np.zeros_like(self.lattice)
        for i in range(1,self.l-1):
            for j in range(1,self.w-1):
                for k in range(1,self.h-1):
                    inst_chance = np.random.rand(1)
                    if inst_chance<= self.frac_charges:
                        charge_dist[i,j,k] = 1
                    else:
                        charge_dist[i,j,k] =0 
        return charge_dist

    def E_field(self):
        """Calculates the electric field from the potential via the equation E = -grad(potential), utilizes the numba function calc_grad3D()"""
        self.Ex,self.Ey,self.Ez = calc_grad3D(self.lattice,1,self.l,self.w,self.h) 
        #Recall electric field is MINUS grad(potential) so now need to make negative
        self.Ex*=-1
        self.Ey*=-1
        self.Ez*=-1
        self.E_lattice = np.sqrt(self.Ex**2 + self.Ey**2 +self.Ez**2)
    
    def Gauss_Seidel_3d(self):
        """Updates a 3d lattice using a gauss-seidel algorithm via a numba function, also saves the previous lattice"""
        self.old_lattice = np.copy(self.lattice)
        self.lattice = gauss_seidel_3d(self.lattice,self.charge_density,self.l,self.w,self.h)
                    
    def GS_SOR_3d(self):
        """Updates a 3d lattice using a successive over-relaxation method using a numba function, also saves previous lattice"""
        self.old_lattice = np.copy(self.lattice)
        self.lattice = gauss_seidel_SOR_3d(self.lattice,self.charge_density,self.l,self.w,self.h,self.sorw)

    def Gauss_Seidel_2d(self):
        """Updates a 2d lattice using a gauss-seidel algorithm via a numba function, also saves the previous lattice"""
        self.old_lattice = np.copy(self.lattice)
        self.lattice = gauss_seidel_2d(self.lattice,self.charge_density,self.l,self.w)

    def GS_SOR_2d(self):
        """Updates a 2d lattice using a successive over-relaxation method using a numba function, also saves previous lattice"""
        self.old_lattice = np.copy(self.lattice)
        self.lattice = gauss_seidel_SOR_2d(self.lattice,self.charge_density,self.l,self.w,self.sorw)

    def update_pot_Jacobi(self):
        """Updates the lattice using a Jacobi algorithm via a numba function, also saves previous lattice"""
        self.old_lattice = np.copy(self.lattice)
        self.lattice = update_poisson_3d(self.lattice,self.charge_density,self.l,self.w,self.h)

    def converg_check(self):
        """Compares the previous potential to the current one, iterates the convergence counter""" 
        self.diff = np.abs(self.old_lattice-self.lattice)/(np.abs(self.lattice)+1e-10)#Confirm this is necessary
        if np.max(self.diff) < self.thresh:
            self.convergence_iter+=1
        else:
            self.convergence_iter=0

    def midplane_cut_pot(self):
        """Takes a midplane slice of the 3d lattice potential to create a visual of the lattice.  This returns plots of all 3 slices (i.e., along each axis)"""

        #X-Y plane cut
        self.cut_xy = self.lattice[:,:,self.h//2] #should select the midline cut of the lattice
        plt.imshow(self.cut_xy, cmap = 'coolwarm')
        plt.colorbar()
        plt.title('X-Y potential plane along midline cut')
        plt.savefig(r'C:\Users\Logan\Downloads\Modelling and Visualization for Physics\Checkpoint 3\Data\Epot_xy.png')
        plt.show()
        plt.close()
        
        #X-Z plane cut
        self.cut_xz = self.lattice[:,self.w//2,:]
        plt.imshow(self.cut_xz, cmap = 'coolwarm')
        plt.colorbar()
        plt.title('X-Z potential plane along midline cut')
        plt.savefig(r'C:\Users\Logan\Downloads\Modelling and Visualization for Physics\Checkpoint 3\Data\Epot_xz.png')
        plt.show()
        plt.close()

        #Y-Z plane cut
        self.cut_yz = self.lattice[self.l//2,:,:]
        plt.imshow(self.cut_yz, cmap = 'coolwarm')
        plt.colorbar()
        plt.title('Y-Z potential plane along midline cut')
        plt.savefig(r'C:\Users\Logan\Downloads\Modelling and Visualization for Physics\Checkpoint 3\Data\Epot_yz.png')
        plt.show()
        plt.close()

    def potential_cuts_run(self): #Works as expected
        """Runs the lattice update method and then returns the midplane cuts"""
        for z in range(1000): #Does 1000 update steps
            self.GS_SOR_3d()#CONFIRM WORKS!!!
            if z%100==0:
                print(f'Step {z} done')
        self.midplane_cut_pot()

    def midplane_cut_Efield(self):
        """Takes a midplane slice of the 3d lattice electric field to create a visual of the lattice.  This returns plots of all 3 slices (i.e., along each axis)"""

        #X-Y cut
        self.cut_xy = self.E_lattice[:,:,self.h//2] #should select the midline cut of the lattice
        plt.imshow(self.cut_xy, cmap = 'coolwarm')
        plt.colorbar()
        plt.title('X-Y Electric Field plane along midline cut')
        plt.savefig(r'C:\Users\Logan\Downloads\Modelling and Visualization for Physics\Checkpoint 3\Data\Efield_xy.png')
        plt.show()
        plt.close()
        
        #X-Z Cut
        self.cut_xz = self.E_lattice[:,self.w//2,:]
        plt.imshow(self.cut_xz, cmap = 'coolwarm')
        plt.colorbar()
        plt.title('X-Z Electric Field along midline cut')
        plt.savefig(r'C:\Users\Logan\Downloads\Modelling and Visualization for Physics\Checkpoint 3\Data\Efield_xz.png')
        plt.show()
        plt.close()
        
        #Y-Z cut
        self.cut_yz = self.E_lattice[self.l//2,:,:]
        plt.imshow(self.cut_yz, cmap = 'coolwarm')
        plt.colorbar()
        plt.title('Y-Z Electric Field plane along midline cut')
        plt.savefig(r'C:\Users\Logan\Downloads\Modelling and Visualization for Physics\Checkpoint 3\Data\Efield_yz.png')
        plt.show()
        plt.close()
    
    def mag_field_calc(self):
        """Calculates the magnetic field via the equation Nabla ^2 A= -mu_0 J """
        #B_x = dA_z/dy
        self.Bx, self.By = mag_cross(self.lattice,self.l,self.w,self.dx)
        self.By*=-1 #Needs negative coefficient
        self.B = np.sqrt(self.Bx**2 + self.By**2) #Calculates the magnitude


    def update_pot_mag_Jacobi(self):
        """Updates the magnetic potential with a jacobi algorithm"""
        self.old_mag_pot = np.copy(self.lattice)
        self.lattice = update_poisson_mag(self.lattice,self.charge_density,self.l,self.w)

    def potential_cuts_mag(self):
        """Updates the lattice a select number of times and then returns a plot of the magnetic potential and the magnetic field strength"""
        # for z in range(1000): #Does 100 update steps
        #     # self.update_pot_mag_Jacobi()#Check works
        #     self.GS_SOR_2d()
        #     if z%100==0:
        #         print(f'Step {z} done')
        # self.mag_field_calc()
        plt.imshow(self.lattice, cmap = 'coolwarm')
        plt.colorbar()
        plt.title(f'Magnetic potential in X-Y plane')
        plt.xlabel(f'X')
        plt.ylabel(f'Y')
        plt.savefig(r'C:\Users\Logan\Downloads\Modelling and Visualization for Physics\Checkpoint 3\Data\PotA.png')
        plt.show()
        plt.close()

        plt.imshow(self.B, cmap = 'coolwarm')
        plt.colorbar()
        plt.title(f'Magnetic field in X-Y plane')
        plt.xlabel(f'X')
        plt.ylabel(f'Y')
        plt.savefig(r'C:\Users\Logan\Downloads\Modelling and Visualization for Physics\Checkpoint 3\Data\BField.png')
        plt.show()
        plt.close()

    def electric_vs_center(self):
        """Calculates the electric field and the potential as a function of distance to the charge.  Does this via taking a slice along the center of the lattice
        Does so via flattening arrays and using masks
        """

        potential_midplane = self.lattice[:,:,self.h//2] #Gets the midplane cut of the potential lattice
        distance_lattice = dist_to_cent(potential_midplane,self.l,self.w)
        self.dist_flat = np.ndarray.flatten(distance_lattice) #Flattens the distance lattice to a 1d array to allow for easier manipulation
        distances = np.unique(self.dist_flat) #Finds the number of unique values in the distance array, since potential is isotropic in space this makes plotting easier
        flat_potential = np.ndarray.flatten(potential_midplane)
        
        flat_Efield = np.ndarray.flatten(self.E_lattice[:,:,self.h//2])

        self.pot_as_dist = []
        self.Efield_as_dist = []
        for r in distances:
                self.pot_as_dist.append(flat_potential[self.dist_flat==r].mean()) #every potential should be the same for this distance, so mean should just return this value
                self.Efield_as_dist.append(flat_Efield[self.dist_flat==r].mean())#See above
        
        #Now plotting potential as function of distance
        plt.figure(figsize=(10,6))
        plt.loglog(distances,self.pot_as_dist)
        plt.xlabel(f'Distance r from charge')
        plt.ylabel(f'Electric Potential $\phi$ at distance r')
        plt.title(f'Log-Log Scaled Potential vs Distance for Monopole in Center')
        plt.show()

        #Now plotting Efield as function of distance
        plt.figure(figsize=(10,6))
        plt.loglog(distances,self.Efield_as_dist)
        plt.xlabel(f'Distance from charge (r)')
        plt.ylabel(f'Electric field magnitude at distance r')
        plt.title(f'Log-Log Scaled Electric Field Strength vs Distance for Monopole in Center')
        plt.show()

    def mag_vs_center(self):
        """Calculates the magnetic field and the potential as a function of distance to the charge. 
         Does so via flattening the array and utilizing masks """
        distance_lattice = dist_to_cent(self.lattice,self.l,self.w) #magnetic lattice is 2d not 3d so we don't need to take any slice
        self.dist_flat = np.ndarray.flatten(distance_lattice) #Flattens the distance lattice to a 1d array to allow for easier manipulation
        distances = np.unique(self.dist_flat) #Finds the number of unique values in the distance array, since potential is isotropic in space this makes plotting easier
        flat_potential = np.ndarray.flatten(self.lattice)
        
        flat_Bfield = np.ndarray.flatten(self.B)

        self.pot_as_dist = []
        self.Bfield_as_dist = []
        for r in distances:
                self.pot_as_dist.append(flat_potential[self.dist_flat==r].mean()) #every potential should be the same for this distance, so mean should just return this value
                self.Bfield_as_dist.append(flat_Bfield[self.dist_flat==r].mean())#See above
        
        #Now plotting potential as function of distance
        plt.figure(figsize=(10,6))
        plt.plot(distances,self.pot_as_dist)
        plt.xscale('log')
        plt.xlabel(f'Logarithm of Distance r from charge')
        plt.ylabel(f'Potential A at distance r')
        plt.title(f'Potential vs Distance for Wire Along Z Axis')
        plt.show()

        #Now plotting Efield as function of distance
        plt.figure(figsize=(10,6))
        plt.loglog(distances,self.Bfield_as_dist)
        plt.xlabel(f'Distance from charge (r)')
        plt.ylabel(f'Magnetic field magnitude at distance r')
        plt.title(f' Log-Log Scaled Magnetic Field Strength vs Distance for Wire Along Z Axis')
        plt.show()

    #The following are all convergence checkers for different update algorithms
    def convergence_test_Jacob3d(self):
        self.iter_step = 0
        while self.convergence_iter<self.criteria:
            if self.iter_step%100==0:
                print(f'On step {self.iter_step}')
            self.update_pot_Jacobi()
            self.converg_check()
            self.iter_step+=1

        print(f'The number of steps needed for iteration is {self.iter_step}')

    def convergence_test_Jacob2d(self):
        self.iter_step = 0
        while self.convergence_iter<self.criteria:
            if self.iter_step%100==0:
                print(f'On step {self.iter_step}')
            self.update_pot_mag_Jacobi()
            self.converg_check()
            self.iter_step+=1
        print(f'The number of steps needed for iteration is {self.iter_step}')

    def convergence_test_GS3d(self):
        self.iter_step = 0
        while self.convergence_iter<self.criteria:
            if self.iter_step%100==0:
                print(f'On step {self.iter_step}')
            self.Gauss_Seidel_3d()
            self.converg_check()
            self.iter_step+=1
        print(f'The number of steps needed for iteration is {self.iter_step}')


    def convergence_test_GS2d(self):
        self.iter_step = 0
        while self.convergence_iter<self.criteria:
            if self.iter_step%100==0:
                print(f'On step {self.iter_step}')
            self.Gauss_Seidel_2d()
            self.converg_check()
            self.iter_step+=1
        print(f'The number of steps needed for iteration is {self.iter_step}')

    def convergence_test_GS_SOR3d(self):
        self.iter_step = 0
        while self.convergence_iter<self.criteria:
            if self.iter_step%100==0:
                print(f'On step {self.iter_step}')
            self.GS_SOR_3d()
            self.converg_check()
            self.iter_step+=1
            if self.iter_step>2000:
                self.convergence_iter=self.criteria #Divergent behavior
        print(f'The number of steps needed for iteration is {self.iter_step}')

    def convergence_test_GS_SOR2d(self):
        self.iter_step = 0
        while self.convergence_iter<self.criteria:
            if self.iter_step%100==0:
                print(f'On step {self.iter_step}')
            self.GS_SOR_2d()
            self.converg_check()
            self.iter_step+=1
        print(f'The number of steps needed for iteration is {self.iter_step}')        
    
    def gen_fig_poisson(self): #Generates an animation for the Poisson Model
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        if self.seed == 4:
            display_data = self.lattice
        else:
            display_data = self.lattice[:, :, self.h//2]
        self.im = self.ax.imshow(display_data, cmap='coolwarm', aspect='auto')
        self.ax.set_title('Poisson Lattice')
        plt.colorbar(self.im, ax=self.ax)

    def update_poisson(self, frame):
        if self.seed == 4:
            self.GS_SOR_2d()
            self.im.set_data(self.lattice)
        else:
            self.GS_SOR_3d()
            self.im.set_data(self.lattice[:, :, self.h//2])
        self.im.set_clim(vmin=self.lattice.min(), vmax=self.lattice.max()) # Dynamically updates colorbar range
        return self.im,

    def animate_poisson(self):
        """Creates an animation for Poisson"""
        self.gen_fig_poisson()
        anim = animation.FuncAnimation(self.fig, self.update_poisson, 100000, repeat=False, interval=1)
        return anim
    
class DataCollection:
    def __init__(self, length,width, height, dx,dt, threshhold, convergence_criteria, start, stop, resolution,filename):
        self.l = length
        self.w = width
        self.h = height
        self.dx = dx
        self.dt = dt
        self.thresh = threshhold
        self.cc = convergence_criteria
        self.start = start
        self.stop = stop
        self.r = resolution
        self.filename = filename

    def sor_trials(self):
        w_values = np.linspace(self.start, self.stop, self.r)
        step_list = []
        
        for w in w_values:
            test = Poisson(self.l,self.w,self.h,1,.5,1,1,self.thresh,self.cc,w)
            test.convergence_test_GS_SOR3d()
            step_list.append(test.iter_step)
            print(f'Done w = {w:.3f}, avg steps = {step_list[-1]:.1f}')
        
        self.data = list(zip(w_values,step_list))
        self.save_data(self.data,headers = ['omega value',f'number of steps till convergence for threshhold = {self.thresh}'])

        plt.figure(figsize=(10,6))
        plt.plot(w_values, step_list, marker='o')
        plt.xlabel('ω')
        plt.ylabel('Average iterations to convergence')
        plt.title('SOR convergence vs relaxation parameter')
        plt.show()
    
    def free_energy_trials(self):
        # Cahn_Hilliard now takes no a,b,k,M arguments
        test = Cahn_Hilliard(self.l, self.w, self.dx, self.dt, self.start)
        test.free_energy_data(self.cc,self.thresh,self.r)
        timelist = range(len(test.free_energy_list))
        self.data =list(zip(timelist,test.free_energy_list))
        self.save_data(self.data,headers=['time', f'free energy for phi0 = {self.start}']) 


    def Electric_field_dist_trials(self):
        test = Poisson(self.l,self.w,self.h,1,.2,1,1,self.thresh,self.cc,self.start)#NOTE, I am using the start value to determine the w 
        test.convergence_test_GS_SOR3d()
        test.E_field() #Calculates the electric field
        test.midplane_cut_Efield()
        test.midplane_cut_pot()
        test.electric_vs_center()

        distances = np.unique(test.dist_flat)  

        self.data = list(zip(distances,test.pot_as_dist,test.Efield_as_dist))
        self.save_data(self.data,headers=['Distances', f'Potential $\phi$ at distance from center', 'Electric field strength at distance from center']) 
    
    def Magnetic_field_dist_trials(self):
        test = Poisson(self.l,self.w,self.h,4,.2,1,1,self.thresh,self.cc,self.start)#NOTE, I am using the start value to determine the w
        test.convergence_test_GS_SOR2d()
        test.mag_field_calc() 
        test.potential_cuts_mag()
        test.mag_vs_center()

        distances = np.unique(test.dist_flat)  

        self.data = list(zip(distances,test.pot_as_dist,test.Bfield_as_dist))
        self.save_data(self.data,headers=['Distances', f'Potential A at distance from center', 'Magnetic field strength at distance from center']) 


    def save_data(self,data,headers):
        with open(self.filename +'.csv', mode = 'w', newline = '', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)
        print (f'Data written to {self.filename}.csv')

    
#Running with dimensionless units: length scale xi=sqrt(k/a), time scale tau=k/(M*a^2)
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--Length',type = int, help = 'Length of the lattice', default=100)
    parser.add_argument('--Width',type = int, help = 'Width of the lattice', default=100)
    parser.add_argument('--Height',type = int, help = 'Height of the lattice', default=100)
    parser.add_argument('--Phi0',type =float , help = 'phi0 value to be used for cahn-hilliard simulation', default=0.0)
    parser.add_argument('--dx',type =float , help = 'Spatial step in units of xi=sqrt(k/a)', default=1.0)
    
    # Permittivity kept for Poisson only
    parser.add_argument('--Perm',type =float , help = 'Permittivity value for Poisson calculations', default=1.0)

    parser.add_argument('--dt',type =float , help = 'Temporal step in units of tau=k/(M*a^2)', default=0.01)
    parser.add_argument('--LatticeType',type =str , help = 'Type of lattice to be used, Cahn_Hilliard or Poisson', default='Cahn_Hilliard')
    parser.add_argument('--Seed',type =int , help = 'Seed corresponding to the specific lattice configuration to be used', default=0)
    parser.add_argument('--FracCharge',type =float , help = 'Fraction of the lattice which are charges', default=0.2)
    parser.add_argument('--w',type =float , help = 'W value to be used in SOR update algorithms', default=1.9)
    parser.add_argument('--Threshhold',type =float , help = 'The threshhold for convergence, the difference between two timesteps defining convergence ', default=1e-6)
    parser.add_argument('--ConvergenceCriteria',type =int , help = 'Number of times the convergence threshhold has to be met before simulation is terminated', default=10)
    parser.add_argument('--Anim',type =bool , help = 'Determines whether or not an animation is created of the system', default=False)
    parser.add_argument('--UpdateRule',type =int , help = 'Determines the update rule to be used for the system: 0 corresponds to Jacobi, 1 to Gauss_Seidel, 2 to ' \
    'SOR Gauss-Seidel', default=0)
    parser.add_argument('--StartVal',type =float , help = 'Determines starting value for any sort of trial run', default=0.0)
    parser.add_argument('--EndVal',type =float , help = 'Determines the ending value for any sort of trial run', default=1.0)
    parser.add_argument('--Resolution',type =int , help = 'The resolution of a trial run, how many steps to increment the trial', default=10)
    parser.add_argument('--Filename',type =str , help = 'Filename to save data to', default='MVP_checkpoint_3.csv')
    parser.add_argument('--Trial',type =bool , help = 'Determines if a data trial is run or a regular run', default=False)
    parser.add_argument('--TrialSeed',type =int , help = 'Seed corresponding to the specific data collection trial to be run' \
    '0 corresponds to a SOR omega variation value trial' \
    '1 corresponds to a Cahn-Hilliard free energy trial' \
    '2 corresponds to a Poisson monopole ' \
    '3 corresponds to a Magnetic Poisson wire configuration', default=0)
    args = parser.parse_args()
    
    if args.Trial ==True:
        #We are running a data collection trial if Trial == True
        if args.TrialSeed ==0:
            SOR_Data = DataCollection(args.Length,args.Width,args.Height,args.dx,args.dt,args.Threshhold,
                                      args.ConvergenceCriteria,args.StartVal,args.EndVal,args.Resolution,args.Filename)
            SOR_Data.sor_trials()

        if args.TrialSeed ==1:
            #RUN WITH dt = 1e-4 TO GET SMOOTH PLOTS
            CH_FreeE_Data = DataCollection(args.Length,args.Width,args.Height,args.dx,args.dt,args.Threshhold,
                                      args.ConvergenceCriteria,args.StartVal,args.EndVal,args.Resolution,args.Filename)
            CH_FreeE_Data.free_energy_trials()
        if args.TrialSeed ==2:
            Monopole_data = DataCollection(args.Length,args.Width,args.Height,args.dx,args.dt,args.Threshhold,
                                      args.ConvergenceCriteria,args.StartVal,args.EndVal,args.Resolution,args.Filename)
            Monopole_data.Electric_field_dist_trials()
        if args.TrialSeed ==3:
            Wire_data = DataCollection(args.Length,args.Width,args.Height,args.dx,args.dt,args.Threshhold,
                                      args.ConvergenceCriteria,args.StartVal,args.EndVal,args.Resolution,args.Filename)
            Wire_data.Magnetic_field_dist_trials()
    else:
        if args.LatticeType == 'Cahn_Hilliard':
            Lattice = Cahn_Hilliard(args.Length, args.Width, args.dx, args.dt, args.Phi0)
            fig = Lattice.animate_fig()
            plt.show()
        else: #create a poisson lattice, run till convergence and print the midplane (if electrical, otherwise just print the lattice)
            Lattice = Poisson(args.Length,args.Width,args.Height,args.Seed,args.FracCharge,args.dx,args.Perm,args.Threshhold,args.ConvergenceCriteria,args.w)
            if args.seed != 4:
                Lattice.convergence_test_GS_SOR3d()
                Lattice.midplane_cut_Efield()
                Lattice.midplane_cut_pot()
            else:
                Lattice.convergence_test_GS_SOR2d()
                Lattice.potential_cuts_mag()