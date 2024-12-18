# import time

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
    
    
