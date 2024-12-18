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
        - data: Data to be plotted
            type: pandas Dataframe
        -magfield: Magnetic field setting of the SE-SPS
            type: float
        - angle: The angle at which the SE-SPS is set to.
            type: float
        - figsize: size of the figure
            type: float
        
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
        - particles: Dataframe containing centroid positions and associated particle label.
            type: pandas DataFrame
        - centroids: The centroids of the clusters.
            type: pandas DataFrame
        - data: The full data set used in clustering.
            type: pandas DataFrame
        
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
            
            
            
        # if self.particles['Particle Label'].str.contains('Protons').any():
        #     Proton_label = self.centroids.loc[self.centroids['Particle Label'] == 'Protons'].index[0]
        #     Protons = self.data[labels == Proton_label]
        #     ProtonGate = self.particle_gate(Protons)
        #     plt.scatter(Protons[X], Protons[Y], marker = '.', color = 'purple', label = 'Protons')
        #     plt.plot(ProtonGate[X], ProtonGate[Y], 'r--', lw=2)
        #     # return ProtonGate
            
        # if self.particles['Particle Label'].str.contains('Tritons').any():
        #     Triton_label = self.centroids.loc[self.centroids['Particle Label'] == 'Tritons'].index[0]
        #     Tritons = self.data[labels == Triton_label]
        #     TritonGate = self.particle_gate(Tritons)
        #     plt.scatter(Tritons[X], Tritons[Y],  marker = '.', label = 'Tritons')
        #     plt.plot(TritonGate[X], TritonGate[Y], 'r--', lw=2)
        #     # return TritonGate
        
        # if self.particles['Particle Label'].str.contains('Deuterons').any():
        #     Deuteron_label = self.centroids.loc[self.centroids['Particle Label'] == 'Deuterons'].index[0]
        #     Deuterons = self.data[labels == Deuteron_label]
        #     DeuteronGate = self.particle_gate(Deuterons)
        #     plt.scatter(Deuterons[X], Deuterons[Y],  marker = '.', label = 'Deuterons')
        #     plt.plot(DeuteronGate[X], DeuteronGate[Y], 'r--', lw=2)
        #     # return DeuteronGate
        
        # if self.particles['Particle Label'].str.contains('Alphas').any():
        #     Alpha_label = self.centroids.loc[self.centroids['Particle Label'] == 'Alphas'].index[0]
        #     Alphas = self.data[labels == Alpha_label]
        #     AlphaGate = self.particle_gate(Alphas)
        #     plt.scatter(Alphas[X], Alphas[Y],  marker = '.', label = 'Alphas')
        #     plt.plot(AlphaGate[X], AlphaGate[Y], 'r--', lw=2)
        #     # return AlphaGate
            
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

    def plot_particle_of_interest(self,labels, particle_of_interest, figsize = (12,12)):
        
        """
        Plots the particle of interest and its associated gate 
        constructed using particle_gate().
        
        Parameters:
        - labels: A list of labels associated to datapoints in data.
            type: ndarray
        - particles: Dataframe containing centroid positions and associated particle label.
            type: pandas DataFrame
        - centroids: The centroids of the clusters.
            type: pandas DataFrame
        - data: The full data set used in clustering.
            type: pandas DataFrame
        - particle_of_interest: The particle you want to gate on.
            type: string
        
        Returns:
        - None
        """
        
        def Outlierplot(self, labels):    
            X = "ScintLeft"; Y = "AnodeBack"
            Outliers = self.data[labels == -1]
            plt.scatter(Outliers[X], Outliers[Y], color = 'grey', marker = '.', label = 'Outliers')
            plt.xlabel('Rest Energy [arb. units]')
            plt.ylabel('Energy Loss [arb. units]')
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(direction = 'in', which = 'both', top =True, right = True)
            plt.legend(loc = 'upper right')
            plt.show()
            
        X = "ScintLeft"; Y = "AnodeBack"
        particle_list = ['Protons', 'Tritons', 'Deuterons', 'Alphas']
        for particle in particle_list:
            if self.particles['Particle Label'].str.contains(particle).any():
                plt.figure(figsize = figsize)
                plt.title(f'{particle} Gate')
                Particle_label = self.centroids.loc[self.centroids['Particle Label'] == particle].index[0]
                Labeled_Particles = self.data[labels == Particle_label]
                ParticleGate = self.particle_gate(Labeled_Particles)
                plt.scatter(Labeled_Particles[X], Labeled_Particles[Y], marker = '.', color = 'purple', label = f'{particle}')
                plt.plot(ParticleGate[1][X], ParticleGate[1][Y], 'r--', lw=2)
                Outlierplot(self.data, labels)
                # return ProtonGate
            else:
                print('There are no {particle}!')
        

    def plot_gate_fulldata(self, labels, full_data, particle_of_interest, figsize = (12,12)):
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
      