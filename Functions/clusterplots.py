from matplotlib.ticker import AutoMinorLocator
from scipy.spatial import ConvexHull

def plot_subsets(data, subsets, magfield, angle, figsize=(12, 12)):
    """
    Plot subsets in a grid of subplots.
    
    Parameters:
    - data: Data to be plotted
        type: pandas Dataframe
    - subsets: list of ndarray, the subsets to plot
        type: list
    -magfield: Magnetic field setting of the SE-SPS
        type: float
    - angle: The angle at which the SE-SPS is set to.
        type: float
    - figsize: size of the figure
        type: float
    
    Returns:
    - None
    """
    SLAB = data
    n_subsets = len(subsets)

    if n_subsets > 1:
        cols = 3  # Number of columns in the subplot grid
        rows = (n_subsets + cols - 1) // cols  # Calculate rows needed
    
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f'Magnetic field: {magfield} kG ; Angle: {angle} degrees')
        axes = axes.flatten()  # Flatten in case of extra empty axes
        
        for i, subset in enumerate(subsets):
            ax = axes[i]
            ax.scatter(subset['ScintLeft'], subset['AnodeBack'], s=10, marker = '.', alpha=0.7)
            # ax.set_title(f"Subset {i+1}")
            # ax.set_xlabel("Feature 1")
            # ax.set_ylabel("Feature 2")
    
        # Hide extra axes
        for j in range(len(subsets), len(axes)):
            axes[j].axis('off')
    
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(1,2, figsize = (10,4), sharey = True)
        fig.suptitle(f'Magnetic field: {magfield} kG ; Angle: {angle} degrees')
        ax[0].hist2d(SLAB['ScintLeft'], SLAB['AnodeBack'], bins = [512,512], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
        ax[0].set_xlabel('Rest Energy [arb. units]', fontsize  = 13)
        ax[0].set_ylabel('Energy Loss [arb. units]', fontsize  = 13)
        ax[0].set_title('Original Data', fontsize  = 13)
        ax[0].tick_params(direction = 'in', which = 'both', top =True, right = True)
        ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax[0].xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax[0].yaxis.set_minor_locator(ticker.MaxNLocator(30))
        ax[0].annotate(f'Total counts: {len(SLAB['ScintLeft'])}', fontsize = 9, xy = [1200,2000])
        
        ax[1].hist2d(subsets[0]['ScintLeft'], subsets[0]['AnodeBack'], bins = [512,512], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
        ax[1].tick_params(direction = 'in', which = 'both', top =True, right = True)
        ax[1].set_xlabel('Rest Energy [arb. units]', fontsize  = 13)
        # ax[1].set_ylabel('Energy Loss [arb. units]')
        ax[1].set_title('Resampled data', fontsize  = 13)
        ax[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins = 7))
        ax[1].xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax[1].annotate(f'Total counts: {len(subsets[0]['ScintLeft'])}', fontsize = 9, xy = [1200,2000])
        plt.subplots_adjust(wspace = 0.0)
        plt.tight_layout()
        plt.savefig('Resampleddata.png')
        
        
 def ParticleGate(particle_pd):
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
    # Filter points for the cluster of interest 
    cluster_points = particle_pd
    
    # Calculate the convex hull
    hull = ConvexHull(cluster_points)
    
    # Extract vertices
    vertices = cluster_points.iloc[hull.vertices]
    # Add first vertex back to close the gate
    vertices = pd.concat([vertices, vertices.iloc[[0]]])
    
    return cluster_points, vertices
     
def plot_particles(labels, particles, centroids, data):
    
    """
    Plots all of the particles identified by the algorithm and the associated 
    gates constructed using ParticleGate().
    
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
    plt.figure()
    if particles['Particle Label'].str.contains(f'Protons').any():
        Proton_label = centroids.loc[centroids['Particle Label'] == 'Protons'].index[0]
        Protons = data[labels == Proton_label]
        ProtonGate = ParticleGate(Protons)
        plt.scatter(Protons[X], Protons[Y], marker = '.', color = 'purple', label = 'Protons')
        plt.plot(ProtonGate[1][X], ProtonGate[1][Y], 'r--', lw=2)
        return ProtonGate
        
    if particles['Particle Label'].str.contains('Tritons').any():
        Triton_label = centroids.loc[centroids['Particle Label'] == 'Tritons'].index[0]
        Tritons = data[labels == Triton_label]
        TritonGate = ParticleGate(Tritons)
        plt.scatter(Tritons[X], Tritons[Y],  marker = '.', label = 'Tritons')
        plt.plot(TritonGate[1][X], TritonGate[1][Y], 'r--', lw=2)
        return TritonGate
    
    if particles['Particle Label'].str.contains('Deuterons').any():
        Deuteron_label = centroids.loc[centroids['Particle Label'] == 'Deuterons'].index[0]
        Deuterons = data[labels == Deuteron_label]
        DeuteronGate =ParticleGate(Deuterons)
        plt.scatter(Deuterons[X], Deuterons[Y],  marker = '.', label = 'Deuterons')
        plt.plot(DeuteronGate[1][X], DeuteronGate[1][Y], 'r--', lw=2)
        return DeuteronGate
    
    if particles['Particle Label'].str.contains('Alphas').any():
        Alpha_label = centroids.loc[centroids['Particle Label'] == 'Alphas'].index[0]
        Alphas = data[labels == Alpha_label]
        AlphaGate = ParticleGate(Alphas)
        plt.scatter(Alphas[X], Alphas[Y],  marker = '.', label = 'Alphas')
        plt.plot(AlphaGate[1][X], AlphaGate[1][Y], 'r--', lw=2)
        return AlphaGate
    
    
    
    Outliers = data[labels == -1]
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

def plot_particle_of_interest(labels, particles, centroids, data, particle_of_interest):
    
    """
    Plots the particle of interest and its associated gate 
    constructed using ParticleGate().
    
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
    
    X = "ScintLeft"; Y = "AnodeBack"
    if particle_of_interest == 'Proton':
        if particles['Particle Label'].str.contains(f'Protons').any():
            plt.figure()
            plt.title(f'{particle_of_interest} Gate')
            Proton_label = centroids.loc[centroids['Particle Label'] == 'Protons'].index[0]
            Protons = data[labels == Proton_label]
            ProtonGate = ParticleGate(Protons)
            plt.scatter(Protons[X], Protons[Y], marker = '.', color = 'purple', label = 'Protons')
            plt.plot(ProtonGate[1][X], ProtonGate[1][Y], 'r--', lw=2)
            Outlierplot(data, labels)
            # return ProtonGate
        else:
            print('There are no protons!')
        
    if particle_of_interest == 'Triton':
        if particles['Particle Label'].str.contains(f'Tritons').any():
            plt.figure()
            plt.title(f'{particle_of_interest} Gate')
            Triton_label = centroids.loc[centroids['Particle Label'] == 'Tritons'].index[0]
            Tritons = data[labels == Triton_label]
            TritonGate = ParticleGate(Tritons)
            plt.scatter(Tritons[X], Tritons[Y],  marker = '.', label = 'Tritons')
            plt.plot(TritonGate[1][X], TritonGate[1][Y], 'r--', lw=2)
            Outlierplot(data, labels)
            # return TritonGate
        else:
            print('There are no tritons!')
    
    if particle_of_interest == 'Deuteron':
        if particles['Particle Label'].str.contains(f'Deuterons').any():
            plt.figure()
            plt.title(f'{particle_of_interest} Gate')
            Deuteron_label = centroids.loc[centroids['Particle Label'] == 'Deuterons'].index[0]
            Deuterons = data[labels == Deuteron_label]
            DeuteronGate =ParticleGate(Deuterons)
            plt.scatter(Deuterons[X], Deuterons[Y],  marker = '.', label = 'Deuterons')
            plt.plot(DeuteronGate[1][X], DeuteronGate[1][Y], 'r--', lw=2)
            Outlierplot(data, labels)
            
        else:
            print('There are no Deuterons!')
    
    if particle_of_interest == 'Alpha':
        if particles['Particle Label'].str.contains(f'Alphas').any():
            plt.figure()
            plt.title(f'{particle_of_interest} Gate')
            Alpha_label = centroids.loc[centroids['Particle Label'] == 'Alphas'].index[0]
            Alphas = data[labels == Alpha_label]
            AlphaGate = ParticleGate(Alphas)
            plt.scatter(Alphas[X], Alphas[Y],  marker = '.', label = 'Alphas')
            plt.plot(AlphaGate[1][X], AlphaGate[1][Y], 'r--', lw=2)
            Outlierplot(data, labels)
            # return AlphaGate
        else:
            print('There are no alphas!')
    
    
def Outlierplot(data, labels):    
    X = "ScintLeft"; Y = "AnodeBack"
    Outliers = data[labels == -1]
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

def plot_gate_fulldata(labels, particles, centroids, data, full_data, particle_of_interest):
    X = "ScintLeft"; Y = "AnodeBack"
    
    if particle_of_interest == 'Proton':
        if particles['Particle Label'].str.contains(f'Protons').any():
            plt.figure()
            plt.title(f'{particle_of_interest} Gate')
            Proton_label = centroids.loc[centroids['Particle Label'] == 'Protons'].index[0]
            Protons = data[labels == Proton_label]
            ProtonGate = ParticleGate(Protons)
            plt.hist2d(full_data[X], full_data[Y], bins = [512,512], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
            # plt.scatter(full_data[X], full_data[Y], marker = '.', color = 'purple', label = 'Protons')
            plt.plot(ProtonGate[1][X], ProtonGate[1][Y], 'r--', lw=2)
            # Outlierplot(data, labels)
            return ProtonGate
        else:
            print('There are no protons!')
        
    if particle_of_interest == 'Triton':
        if particles['Particle Label'].str.contains(f'Tritons').any():
            plt.figure()
            plt.title(f'{particle_of_interest} Gate')
            Triton_label = centroids.loc[centroids['Particle Label'] == 'Tritons'].index[0]
            Tritons = data[labels == Triton_label]
            TritonGate = ParticleGate(Tritons)
            plt.hist2d(full_data[X], full_data[Y], bins = [512,512], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
            # plt.scatter(full_data[X], full_data[Y],  marker = '.', label = 'Tritons')
            plt.plot(TritonGate[1][X], TritonGate[1][Y], 'r--', lw=2)
            # Outlierplot(data, labels)
            return TritonGate
        else:
            print('There are no tritons!')
    
    if particle_of_interest == 'Deuteron':
        if particles['Particle Label'].str.contains(f'Deuterons').any():
            plt.figure()
            plt.title(f'{particle_of_interest} Gate')
            Deuteron_label = centroids.loc[centroids['Particle Label'] == 'Deuterons'].index[0]
            Deuterons = data[labels == Deuteron_label]
            DeuteronGate =ParticleGate(Deuterons)
            plt.hist2d(full_data[X], full_data[Y], bins = [512,512], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
            # plt.scatter(full_data[X], full_data[Y],  marker = '.', label = 'Deuterons')
            plt.plot(DeuteronGate[1][X], DeuteronGate[1][Y], 'r--', lw=2)
            # # Outlierplot(data, labels)
            return DeuteronGate
        else:
            print('There are no Deuterons!')
    
    if particle_of_interest == 'Alpha':
        if particles['Particle Label'].str.contains(f'Alphas').any():
            plt.figure()
            plt.title(f'{particle_of_interest} Gate')
            Alpha_label = centroids.loc[centroids['Particle Label'] == 'Alphas'].index[0]
            Alphas = data[labels == Alpha_label]
            AlphaGate = ParticleGate(Alphas)
            plt.hist2d(full_data[X], full_data[Y], bins = [512,512], range = [[0,2000],[0,2500]],cmap = 'viridis', norm = colors.LogNorm(), alpha = 0.8)
            plt.plot(AlphaGate[1][X], AlphaGate[1][Y], 'r--', lw=2)
            # Outlierplot(data, labels)
            return AlphaGate
        else:
            print('There are no alphas!')
