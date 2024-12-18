# import time
import glob
import numpy as np
import pandas as pd
import os

from skopt import gp_minimize
from sklearn.cluster import HDBSCAN

from get_particle_class import GetParticleClass
from clusterplots import ClusterPlots
from density_aware_resample import density_aware_resample



import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)



class PIDClassifier:
    def __init__(self,
                 particle_of_interest, 
                 magfield: float, 
                 angle: float, 
                 datapath, 
                 n_subset_samples,
                 k_NearestNeighbors,
                 prange = [(15, 40),(20, 70)],
                 outputpath = './',
                 showplots = True,
                 resample = True,
                 savefile = False
                 ):
        """
         Run entirety of Clustering algorithm to identify and classify
         clusters from PID plots.
         
         Parameters:
         - particle_of_interest: the particle you want to gate on for analysis
             type: string
             options: 'Proton', 'Triton', 'Deuteron', 'Alpha'
             
         - magfield: The magnetic field setting of the SPS in kG
             type: float
             
         - angle: The angle at which the SE-SPS is set at.
             type: float
             
         - datapath: Path to PID data files
             type: string
             
         - n_subset_samples: The number of resampled datasets to create, default is 1.
             type: int
             comments: Currently, the algorithm only takes one.
             
         - k_NearestNeighborhoods: Number of neighbors to use by default for kneighbors queries used in density_aware_resample function.
             type: int
        
         - prange: The range for min_samples and min_cluster_size parameters for HDBSCAN
             type: list of ints with shape (2,2)
        
         - outputpath: The path for the outputted file containing the gate's vertices to be put.
             type: string
        
         - resample: Gives the option to resample data for more efficiency.
             type: boolean
             options: True, False
             
       
         - savefile: Determines whether the gate's vertices will be outputed to a txt file.
             type: boolean
             options: True, False
             
         Returns:
         - None if savefile = False
         - Particle gate vertices if savefile = True
         
        """        
        self.particle_of_interest = particle_of_interest
        self.magfield = magfield
        self.angle = angle
        self.datapath = datapath
        self.n_subset_samples = n_subset_samples
        self.k_NearestNeighbors = k_NearestNeighbors
        self.prange = prange
        self.outputpath = outputpath
        self.showplots = showplots
        self.resample = resample
        self.savefile = savefile        
        
        self.resampled_data = None
        self.particles = None
        self.centroids = None
        self.labels = None
        
    def load_data(self):
        """
        This function loads data located in the datapath. These need to be in
        txt files, but should be updated to include ROOT files.
        
        """
        runs = sorted(glob.glob(os.path.join(self.datapath, '*.txt')))
        X = "ScintLeft"; Y = "AnodeBack"
        datalist = [pd.DataFrame(np.loadtxt(run, unpack = False), columns = [X,Y]) for run in runs]       
        return datalist
    
    def cleandata(self):
        """
        This function loads the data and trims the ranges so that some noise is cut.
        
        """
        SLABdat = self.load_data()
        
        X = "ScintLeft"; Y = "AnodeBack"
        SL_min = 0; SL_max = 2000
        AB_min = 0 ; AB_max = 4000 
        
        
        SLAB_full = pd.concat(SLABdat, axis = 0)
        
            
        # SLAB_full = SLABdat[0]
        SLAB = SLAB_full[(SLAB_full[X] >= SL_min) & (SLAB_full[X] <= SL_max) & (SLAB_full[Y] >= AB_min) & (SLAB_full[Y] <= AB_max)]
        SLAB = SLAB.reset_index(drop=True)# pandas keeps the indices of the original, we want to reset to avoid any future issues
       
        return SLAB
    
    def resample_data(self):
        """
        Resamples dataset to make HDBSCAN optmizition and implementation
        more efficient. 

        Returns
        -------
        newdata : the data set to be used by other functions. If the original dataset
        contains more than 10,000 data points, then it resamples using the 
        density_aware_resample function.         

        """
        X = "ScintLeft"; Y = "AnodeBack"
        
        if self.resample == True:
       
        
            sample_k = self.k_NearestNeighbors 
            n_samples_per_subset = self.n_subset_samples
            
            SLAB = self.cleandata()
            if len(SLAB) > 10000:
                
                subsets = density_aware_resample(SLAB, # the original dataset
                                                 n_samples = n_samples_per_subset, # How many data points in new set
                                                 k = sample_k # Number of neighbors for NearestNeighbors
                                                 ) 
                
                newdata = pd.DataFrame(subsets[0], columns = [X,Y])
                print('Using the resampled data, check plot if it is admissable.')
                
            else:
            
                print('Too small to resample')
                newdata = SLAB
                
        else:
            newdata = self.cleandata()
         
        return newdata
    
    def optimize_hdbscan(self):
        
        """
        Function that uses scikit-optimize's gp_minimize function to perform
        Bayesian optimization of the HDBSCAN parameters.
        
        Returns:
        - result.x: The best parameters values for min_samples and min_cluster_size
        which is used in the final implementation of HDBSCAN.
        
        """
        
        def clustering_score(params):
            min_samples, min_cluster_size = params
            db = HDBSCAN(min_samples=int(min_samples), min_cluster_size=int(min_cluster_size))
            clusterer = db.fit(self.resampled_data)
            labels = clusterer.labels_
            # penalize degenerate clustering 
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1 and n_clusters < 5:  # Between 2 to 4 clusters            
                return 1.0 - np.mean(clusterer.probabilities_)
            return 1.0  # Penalize bad clustering
        
        print('Optimizing hyperparameters for HDBSCAN...')
        result = gp_minimize(clustering_score, self.prange, n_calls=50, random_state=42)
        print(f"Optimization complete! \n Best parameters for HDBSCAN: \n min_samples: {result.x[0]} \n min_cluster_size: {result.x[1]}")
        return result.x
    
    def perform_hdbscan(self):
        """
        

        Returns
        -------
        labels : ndarray
            Array of label values from -1 (outliers) to n-1 where n = # of 
            clusters identified
        centroids : ndarray
            Array of centroid X and Y positions 
        particles : list
            List of particle names associated to the centroid position given.

        """
        
        min_samples, min_cluster_size = self.optimize_hdbscan()
        # data = self.resampled_data
        db = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size).fit(self.resampled_data)
        labels = db.labels_
        
        # Number of clusters in labels, ignoring outliers
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Number of Clusters:', n_clusters_)

        # Getting cluster centroids
        unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
        centroids = pd.DataFrame(np.array([self.resampled_data[labels == label].mean(axis=0) for label in unique_labels]), columns = ['Xpos', 'Ypos'])
        
        print('Assigning particle classes to clusters...')
        Classifier = GetParticleClass(centroids)
        particles = Classifier.classify_particle()
        print(f'Cluster centroids and labels: \n {particles}')
              
        return labels, centroids, particles
    
    def get_particle_gates(self):
        
        """
        Plots all of the particles identified by the algorithm and the associated 
        gates constructed using particle_gate().
        
        If savefile == True, then it saves the desired particle of interest's 
        gate vertices to a file in the inputted outputpath with the following 
        syntax: {particle_of_interest}cut_{magfield}kG_{angle}deg.txt.
        
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
  
        self.labels, self.centroids, self.particles = self.perform_hdbscan()

        Plots = ClusterPlots(self.resampled_data, 
                             self.particles, 
                             self.centroids,
                             self.magfield,
                             self.angle
                             )
        
        allparts = []; alllabels = []; gates = []
        particle_list = ['Protons', 'Tritons', 'Deuterons', 'Alphas']
        for particle in particle_list:
            if self.particles['Particle Label'].str.contains(particle).any():
                Part_label = self.centroids.loc[self.centroids['Particle Label'] == particle].index[0]
                Parts = self.resampled_data[self.labels == Part_label]
                PartGate = Plots.particle_gate(Parts)
                allparts.append(Parts)
                alllabels.append(particle)
                gates.append(PartGate)
              
                
              # Saving the particle of interest's gate vertices to text file
              # Changes can be made to save this to a TCUTG file in ROOT.
                if particle.startswith(self.particle_of_interest) and self.savefile:
                        filename = f"{self.particle_of_interest}Cut_{self.magfield}kG_{self.angle}deg.txt"
                        print(f'Saving gate vertices to {filename}...')
                        np.savetxt(os.path.join(self.outputpath, filename), PartGate, fmt='%.2f')

            else:
                print(f'There are no {particle}!')
                

        return allparts, alllabels, gates


    def run(self):
        
        """
        Runs the full PID classification process.
        
        Returns:
        - particle_groups: list
            The data within the particle gates.
            
        - particle_labels: list
            List of particles that have been assigned.
        
        - particle_gates: list
            List of vertices for each particle gate constructed.
            
        """
        
        
        self.resampled_data = self.resample_data()
        particle_groups, particle_labels, particle_gates = self.get_particle_gates()
        
        Plots = ClusterPlots(self.resampled_data, 
                             self.particles, 
                             self.centroids,
                             self.magfield,
                             self.angle,
                             "ScintLeft", 
                             "AnodeBack")
        
        if self.showplots:
        
            Plots.plotsubsets(self.cleandata(), figsize = (12,6))
            
            Plots.plot_particles(labels = self.labels, figsize = (8,6))
            
            Plots.plot_gate_fulldata(particle_of_interest = self.particle_of_interest,
                                     labels = self.labels,
                                     full_data = self.cleandata(),
                                     figsize = (8,6)                             
                                     )
        

        return particle_groups, particle_labels, particle_gates;
   

path = './data/9kG/'
test = PIDClassifier(particle_of_interest= 'Triton',
              magfield = 10, 
              angle = 35,
              datapath = path, 
              n_subset_samples = 4000,  
              k_NearestNeighbors = 5, 
              prange = [(15, 40),(20, 70)],
              outputpath= './cuts',
              showplots = True,
              resample = True,
              savefile = True).run()
