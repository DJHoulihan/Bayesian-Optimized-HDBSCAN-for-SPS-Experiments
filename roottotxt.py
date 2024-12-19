import ROOT
import glob
import os
import numpy as np
import pandas as pd
'''
This file takes root files from an SPS experiment, converts the scintLeft (rest energy) and anodeBack (energy loss) data to panda dataframes, and saves the data to 
.txt files.
'''
def roottotxt(root_files_dir, output_dir):
    # Set the directory containing the ROOT files
    root_files_dir = root_files_dir  # Replace with the path to your directory containing ROOT files
    output_dir = "./SLABtxt/"  # Directory to save output .txt files

    # Get all the ROOT files in the directory (adjust the extension if necessary)
    root_files = glob.glob(os.path.join(root_files_dir, "*.root"))

    # Loop through each ROOT file and extract data
    for root_file_path in root_files:
        # Get the base name of the file (without extension) to use in output file names
        base_name = os.path.basename(root_file_path).replace(".root", "")
        
        # Open each ROOT file
        root_file = ROOT.RDataFrame("SPSTree", root_file_path)
        PID_filter_cond = f"anodeBackTime != -1e6 && xavg != -1e6 "
        root_file_filtered = root_file.Filter(f"{PID_filter_cond}")
        # tree = root_file.Get("SPSTree")  # Replace with your TTree name
        SL = pd.DataFrame(root_file_filtered.AsNumpy(columns = ["scintLeft"]))
        AB = pd.DataFrame(root_file_filtered.AsNumpy(columns = ["anodeBack"]))

        datatosave = np.column_stack((SL,AB))
        
        np.savetxt(f"{output_dir}{base_name}.txt", datatosave, header = "scintLeft\tanodeBack", delimiter = "\t")

        # root_file.Close()

    print("Processing complete!")
    return None
