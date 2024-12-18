# import time
import glob
import numpy as np
import pandas as pd
import os

from skopt import gp_minimize
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
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
        self.resample = resample
        self.savefile = savefile        
        
        self.resampled_data = None
        self.particles = None
        self.centroids = None
        self.labels = None
        
    def load_data(self):
        runs = sorted(glob.glob(os.path.join(self.datapath, '*.txt')))
        X = "ScintLeft"; Y = "AnodeBack"
        datalist = [pd.DataFrame(np.loadtxt(run, unpack = False), columns = [X,Y]) for run in runs]       
        return datalist
    
    def cleandata(self):
    
        SLABdat = self.load_data()
        
        X = "ScintLeft"; Y = "AnodeBack"
        SL_min = 0; SL_max = 2000
        AB_min = 0 ; AB_max = 4000 
        
        
        SLAB_full = pd.concat(SLABdat, axis = 0)
        
            
        # SLAB_full = SLABdat[0]
        SLAB = SLAB_full[(SLAB_full[X] >= SL_min) & (SLAB_full[X] <= SL_max) & (SLAB_full[Y] >= AB_min) & (SLAB_full[Y] <= AB_max)]
        SLAB = SLAB.reset_index(drop=True)# pandas keeps the indices of the original, we want to reset to avoid any future issues
        # print(f"Data type: {type(SLAB)}")
        return SLAB
    
    def resample_data(self):
        
        X = "ScintLeft"; Y = "AnodeBack"
        if self.resample == True:
       
        # 
        #                           Resampling section
        #
        
            sample_k = self.k_NearestNeighbors 
            n_samples_per_subset = self.n_subset_samples
            SLAB = self.cleandata()
            
            if len(SLAB) > 10000:
                
                subsets = density_aware_resample(SLAB, 
                                                 n_samples = n_samples_per_subset,
                                                 k = sample_k)
                
                # ClusterPlots.plotsubsets(SLAB, subsets, self.magfield, self.angle)
                newdata = pd.DataFrame(subsets[0], columns = [X,Y])
                print('Using the resampled data, check plot if it is admissable.')
                
            else:
                
                print('Too small to resample')
                plt.figure()
                plt.hist2d(SLAB[X], SLAB[Y], bins = [256,256], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
                plt.xlabel('Rest Energy [arb. units]')
                plt.ylabel('Energy Loss [arb. units]')
                plt.annotate(f'Total counts: {len(SLAB['ScintLeft'])}', fontsize = 9, xy = [1200,2000])
                ax = plt.gca()
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
                ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
                ax.yaxis.set_minor_locator(ticker.MaxNLocator(30))
                newdata = SLAB
        else:
            
            plt.figure()
            plt.hist2d(SLAB[X], SLAB[Y], bins = [256,256], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
            plt.xlabel('Rest Energy [arb. units]')
            plt.ylabel('Energy Loss [arb. units]')
            plt.annotate(f'Total counts: {len(SLAB['ScintLeft'])}', fontsize = 9, xy = [1200,2000])
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
            ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
            ax.yaxis.set_minor_locator(ticker.MaxNLocator(30))
            newdata = SLAB
            
        return newdata
    
    def optimize_hdbscan(self):
        
        # self.resampled_data = self.resample_data(prints = True)
        
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
        # self.resampled_data = self.resample_data(prints = True)
        self.labels, self.centroids, self.particles = self.perform_hdbscan()
        
        # print(self.resampled_data)
        # print(f"Labels: {self.labels}")
        # print(f"Centroids: {self.centroids}")
        # print(f"Particles: {self.particles}")
        Plots = ClusterPlots(self.resampled_data, 
                             self.particles, 
                             self.centroids,
                             self.magfield,
                             self.angle
                             )
        
        allparts = []; alllabels = []; gates = []
        if self.particles['Particle Label'].str.contains('Protons').any():
            Proton_label = self.centroids.loc[self.centroids['Particle Label'] == 'Protons'].index[0]
            # print(f"type(self.labels): {type(self.labels)}")
            # print(f"type(Proton_label): {type(Proton_label)}")
            Protons = self.resampled_data[self.labels == Proton_label]
            ProtonGate = Plots.particle_gate(Protons)
            allparts.append(Protons)
            alllabels.append('Protons')
            gates.append(ProtonGate)
            # return ProtonGate
            
        if self.particles['Particle Label'].str.contains('Tritons').any():
            Triton_label = self.centroids.loc[self.centroids['Particle Label'] == 'Tritons'].index[0]
            Tritons = self.resampled_data[self.labels == Triton_label]
            TritonGate = Plots.particle_gate(Tritons)
            allparts.append(Tritons)
            alllabels.append('Tritons')
            gates.append(TritonGate)
            # return TritonGate
        
        if self.particles['Particle Label'].str.contains('Deuterons').any():
            Deuteron_label = self.centroids.loc[self.centroids['Particle Label'] == 'Deuterons'].index[0]
            Deuterons = self.resampled_data[self.labels == Deuteron_label]
            DeuteronGate = Plots.particle_gate(Deuterons)
            allparts.append(Deuterons)
            alllabels.append('Deuterons')
            gates.append(DeuteronGate)
            # return DeuteronGate
        
        if self.particles['Particle Label'].str.contains('Alphas').any():
            Alpha_label = self.centroids.loc[self.centroids['Particle Label'] == 'Alphas'].index[0]
            Alphas = self.resampled_data[self.labels == Alpha_label]
            AlphaGate = Plots.particle_gate(Alphas)
            allparts.append(Alphas)
            alllabels.append('Alphas')
            gates.append(AlphaGate)
            # return AlphaGate
        
        return allparts, alllabels, gates
    
        
    def save_gate(self, gate_vertices):
        if self.savefile:
            filename = f"{self.particle_of_interest}cut_{self.magfield}kG_{self.angle}deg.txt"
            print(f'Saving gate vertices to {filename}...')
            np.savetxt(os.path.join(self.outputpath, filename), gate_vertices, fmt='%.2f')
        return None

    def run(self):
        
        """
        Run the full PID classification process.
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
        
        Plots.plotsubsets(self.cleandata(), figsize = (12,6))
        
        Plots.plot_particles(labels = self.labels, figsize = (8,6))
        
        Plots.plot_gate_fulldata(particle_of_interest = self.particle_of_interest,
                                 labels = self.labels,
                                 full_data = self.cleandata(),
                                 figsize = (8,6)                             
                                 )
        
        
        
        # Plotting and gate extraction omitted for brevity
        self.save_gate(particle_gates[1])  # Save centroids as a placeholder
        

        return particle_groups, particle_labels, particle_gates;
    # partgroups, partlabels, partgates = plot_particles(labels, particles , centroids , data)

path = './data/7.3kG/'
test = PIDClassifier(particle_of_interest= 'Deuteron',
              magfield = 10, 
              angle = 35,
              datapath = path, 
              n_subset_samples = 4000,  
              k_NearestNeighbors = 5, 
              prange = [(15, 40),(20, 70)],
              outputpath= './',
              resample = True,
              savefile = False).run()


'''
    
 
def PIDClassifier(particle_of_interest, 
                  magfield: float, 
                  angle: float, 
                  datapath, 
                  n_subset_samples,
                  k_NearestNeighbors,
                  prange = [(15, 40),(20, 70)],
                  outputpath = './',
                  resample = True,
                  multfiles = False,
                  savefile = False):
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
    
    
    # time.sleep(2)
    SLABdat = load_data(path = datapath)
    
    X = "ScintLeft"; Y = "AnodeBack"
    SL_min = 0; SL_max = 2000
    AB_min = 0 ; AB_max = 4000 
    
    if multfiles == True:
        SLAB_full = pd.concat(SLABdat, axis = 0)
    else:
        SLAB_full = SLABdat[0]
        
    # SLAB_full = SLABdat[0]
    SLAB = SLAB_full[(SLAB_full[X] >= SL_min) & (SLAB_full[X] <= SL_max) & (SLAB_full[Y] >= AB_min) & (SLAB_full[Y] <= AB_max)]
    SLAB = SLAB.reset_index(drop=True)# pandas keeps the indices of the original, we want to reset to avoid any future issues
    
    if resample == True:
    # 
    #                           Resampling section
    #
    
        n_subsets = 1
        sample_k = k_NearestNeighbors 
        n_samples_per_subset = n_subset_samples
        # print(n_samples_per_subset)
        
        if len(SLAB) > 3000:
            subsets, subsets_unscaled = density_aware_resample(SLAB, n_samples=n_samples_per_subset, n_subsets=n_subsets, k = sample_k)
            plot_subsets(SLAB, subsets_unscaled, magfield, angle)
            data = pd.DataFrame(subsets_unscaled[0], columns = [X,Y])
            print('Using the resampled data, check plot if it is admissable.')
        else:
            print('Too small to resample')
            plt.figure()
            plt.hist2d(SLAB[X], SLAB[Y], bins = [256,256], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
            plt.xlabel('Rest Energy [arb. units]')
            plt.ylabel('Energy Loss [arb. units]')
            plt.annotate(f'Total counts: {len(SLAB['ScintLeft'])}', fontsize = 9, xy = [1200,2000])
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
            ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
            ax.yaxis.set_minor_locator(ticker.MaxNLocator(30))
            data = SLAB
    else:
        data = SLAB
        plt.figure()
        plt.hist2d(SLAB[X], SLAB[Y], bins = [256,256], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
        plt.xlabel('Rest Energy [arb. units]')
        plt.ylabel('Energy Loss [arb. units]')
        plt.annotate(f'Total counts: {len(SLAB['ScintLeft'])}', fontsize = 9, xy = [1200,2000])
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax.yaxis.set_minor_locator(ticker.MaxNLocator(30))
    

    #
    #         Optimizing hyperparameters for HDBSCAN
    #
    #   prange = [(15, 40),  # min_samples
    #           (20, 70)]  # min_cluster_size
    def clustering_score(params):
        min_samples, min_cluster_size = params
        db = HDBSCAN(min_samples=int(min_samples), min_cluster_size=int(min_cluster_size))
        clusterer = db.fit(data)
        labels = clusterer.labels_
        # penalize degenerate clustering 
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1 and n_clusters < 5:  # Between 2 to 4 clusters            
            return 1.0 - np.mean(clusterer.probabilities_)
        return 1.0  # Penalize bad clustering
    
    print('Optimizing hyperparameters for HDBSCAN...')
    result = gp_minimize( clustering_score, prange, n_calls=50, random_state=42)
    print(f"Optimization complete! \n Best parameters for HDBSCAN: \n min_samples: {result.x[0]} \n min_cluster_size: {result.x[1]}")
    
    #
    #           Performing HDBSCAN clusteering algortihm
    # - Provides cluster labels and gets cluster centroid locations
    #
    print('Performing HDBSCAN clustering algorithm...')
    db = HDBSCAN(min_samples = result.x[0], min_cluster_size = result.x[1]).fit(data)
    labels = db.labels_ # getting labels of clusters
    print('HDBSCAN algorithm complete!')
    
    # Number of clusters in labels, ignoring outliers
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Number of Clusters:', n_clusters_)

    # Getting cluster centroids
    unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
    centroids = pd.DataFrame(np.array([data[labels == label].mean(axis=0) for label in unique_labels]), columns = ['Xpos', 'Ypos'])

    #
    #          Prescribe the cluster centroids to particle classes
    #
    print('Assigning particle classes to clusters...')
    particles = get_particle_class(centroids)
    print(f'Cluster centroids and labels: \n {particles}')

    # partgroups = the datapoints within each group
    # partlabels = the particle labels for each group
    # partgates = the vertices of each particle gate
    partgroups, partlabels, partgates = plot_particles(labels, particles , centroids , data)
    
    # plot_particles(labels, particles , centroids , data)
    print('Check the gates and see if they are valid.')
    
    plotofinterest = plot_particle_of_interest(labels, particles, centroids, data, f'{particle_of_interest}')
    print('Checking gates with full dataset')
    # plot_gate_fulldata(labels, particles, centroids, data, SLAB, f'{particle_of_interest}')
        # print(plotofinterest)
    # return plotofinterest
    
    if savefile == True:
        filename = f'{particle_of_interest}cut_{magfield}kG_{angle}deg.txt'
        print('Saving vertices to file name:', filename)
        np.savetxt(os.path.join(outputpath, filename), plotofinterest[1], fmt='%.2f')
        # df.to_csv('output.txt', sep='\t', index=False)
    else:
        None
    return partgroups, partlabels, partgates
    
'''
