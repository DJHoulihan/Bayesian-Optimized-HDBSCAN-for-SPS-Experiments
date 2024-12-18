from matplotlib.ticker import AutoMinorLocator
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
import pandas as pd


class ClusterPlots:
    def __init__(self,
                 data,
                 particles,
                 centroids,
                 magfield,
                 angle,
                 X = "ScintLeft", 
                 Y = "AnodeBack",
                 ):
        
        self.data = data
        self.particles = particles
        self.centroids = centroids
        self.magfield = magfield
        self.angle = angle
        self.X = X
        self.Y = Y
        
        
        
    def plotsubsets(self, full_data, figsize = (12,12)):
        """
        Plot subsets in a grid of subplots.
        
        Parameters:
        - fulldata: The full dataset inputted into the algorithm.
            type: Pandas DataFrame
        - figsize: size of the figure
            type: tuple
        
        Returns:
        - None
        """
        
        fig, ax = plt.subplots(1,2, figsize = figsize, sharey = True)
        fig.suptitle(f'Magnetic field: {self.magfield} kG ; Angle: {self.angle} degrees')
        ax[0].hist2d(full_data['ScintLeft'], full_data['AnodeBack'], bins = [512,512],
                     range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
        ax[0].set_xlabel('Rest Energy [arb. units]', fontsize  = 13)
        ax[0].set_ylabel('Energy Loss [arb. units]', fontsize  = 13)
        ax[0].set_title('Original Data', fontsize  = 13)
        ax[0].tick_params(direction = 'in', which = 'both', top =True, right = True)
        ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax[0].xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax[0].yaxis.set_minor_locator(ticker.MaxNLocator(30))
        ax[0].annotate(f"Total counts: {len(full_data['ScintLeft'])}", fontsize = 9, xy = [1100,2000])
        
        ax[1].hist2d(self.data['ScintLeft'], self.data['AnodeBack'], bins = [512,512], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
        ax[1].tick_params(direction = 'in', which = 'both', top =True, right = True)
        ax[1].set_xlabel('Rest Energy [arb. units]', fontsize  = 13)
        # ax[1].set_ylabel('Energy Loss [arb. units]')
        ax[1].set_title('Resampled data', fontsize  = 13)
        ax[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax[1].xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax[1].annotate(f"Total counts: {len(self.data['ScintLeft'])}", fontsize = 9, xy = [1200,2000])
        plt.subplots_adjust(wspace = 0.0)
        plt.show()
        
        return None
    
    def particle_gate(self,particle_pd):
        """
        Finds the particle group of interest, calculates the Convex hull,
        and extracts vertices of the particle group.
        
        Parameters:
        - particle_pd: Dataframe of data points corresponding to a cluster group.
            type: pandas DataFrame
        
        Returns:
        - cluster_points: The data points of the group of interest.
            type: pandas DataFrame
        - vertices: The vertices of the group of interest.
            type: pandas DataFrame
        """
        
        hull = ConvexHull(particle_pd[[self.X, self.Y]])
        vertices = particle_pd.iloc[hull.vertices]
        vertices = pd.concat([vertices, vertices.iloc[[0]]])
        
        return vertices
    
    
    def plot_particles(self, labels, figsize = (12,12)):
        
        """
        Plots all of the particles identified by the algorithm and the associated 
        gates constructed using particle_gate().
        
        Parameters:
        - labels: A list of labels associated to datapoints in data.
            type: ndarray
        - figsize: The size of the figure (length, height)
            type: tuple
        
        Returns:
        - None
        """
        
        X = "ScintLeft"; Y = "AnodeBack"
        plt.figure(figsize = figsize)
        
        X = "ScintLeft"; Y = "AnodeBack"
        particle_list = ['Protons', 'Tritons', 'Deuterons', 'Alphas']
        for particle in particle_list:
            if self.particles['Particle Label'].str.contains(particle).any():
                Part_label = self.centroids.loc[self.centroids['Particle Label'] == particle].index[0]
                Parts = self.data[labels == Part_label]
                PartGate = self.particle_gate(Parts)
                plt.scatter(Parts[X], Parts[Y], marker = '.', label = f'{particle}')
                plt.plot(PartGate[X], PartGate[Y], 'r--', lw=2)
            
        Outliers = self.data[labels == -1]
        plt.scatter(Outliers[X], Outliers[Y], color = 'grey', marker = '.', label = 'Outliers')
        plt.xlabel('Rest Energy [arb. units]')
        plt.ylabel('Energy Loss [arb. units]')
        plt.title(f'Particle Gates for B = {self.magfield}, $\\theta$ = {self.angle}')
        plt.ylim((0,2500))
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction = 'in', which = 'both', top =True, right = True)
        plt.legend(loc = 'upper right')
        plt.show()
        
        return None

    def plot_gate_fulldata(self, labels, full_data, particle_of_interest, figsize = (12,12)):
        """
        Plots the full data set and the particle of interest's gate.
        
        Parameters:
        - labels: A list of labels associated to datapoints in data.
            type: ndarray
        - fulldata: The full dataset inputted into the algorithm.
            type: Pandas DataFrame
        - particle_of_interest: Name of particle you want to gate around.
            type: String
            options: 'Proton', 'Triton', 'Deuteron', 'Alpha'
        - figsize: The size of the figure (length, height)
            type: tuple        
        
        Returns:
        - None
        """
        
        X = "ScintLeft"; Y = "AnodeBack"
        
        
        particle_list = ['Protons', 'Tritons', 'Deuterons', 'Alphas']
       
        
        for particle in particle_list:
            if particle.startswith(particle_of_interest) and self.particles['Particle Label'].str.contains(particle).any():
                plt.figure(figsize = figsize)
                plt.title(f'{particle_of_interest} Gates for B = {self.magfield}, $\\theta$ = {self.angle}')
                plt.xlabel('Rest Energy [arb. units]')
                plt.ylabel('Energy Loss [arb. units]')
                
                Particle_label = self.centroids.loc[self.centroids['Particle Label'] == particle].index[0]
                Labeled_Particles = self.data[labels == Particle_label]
                ParticleGate = self.particle_gate(Labeled_Particles)
                # plt.scatter(Labeled_Particles[X], Labeled_Particles[Y], marker = '.', color = 'purple', label = f'{particle}')
                plt.hist2d(full_data[X], full_data[Y], bins = [512,512], range = [[0,2000],[0,2500]],
                           cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
                plt.plot(ParticleGate[X], ParticleGate[Y], 'r--', lw=2)
                
                ax = plt.gca()
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(direction = 'in', which = 'both', top =True, right = True)
                
                
            else:
                None
      
