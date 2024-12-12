def load_data(path):
    runs = sorted(glob.glob(os.path.join(path, '*.txt')))
    X = "ScintLeft"; Y = "AnodeBack"
    datalist = [pd.DataFrame(np.loadtxt(run, unpack = False), columns = [X,Y]) for run in runs]       
    return datalist
