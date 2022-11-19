import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
import numpy as np
import os
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import _CNN
from data_util import split_csv
from data_util import CNN_Data
from data_util import brain_regions


# This is a color map that you can use to plot the SHAP heatmap on the input MRI
colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


# Returns two data loaders (objects of the class: torch.utils.data.DataLoader) that are
# used to load the background and test datasets.
def prepare_dataloaders(bg_csv, test_csv, bg_batch_size = 8, test_batch_size= 1, num_workers=1):
    '''
    Attributes:
        bg_csv (str): The path to the background CSV file.
        test_csv (str): The path to the test data CSV file.
        bg_batch_size (int): The batch size of the background data loader
        test_batch_size (int): The batch size of the test data loader
        num_workers (int): The number of sub-processes to use for dataloader
    '''
    bg_dataset = CNN_Data(bg_csv)
    test_dataset = CNN_Data(test_csv)

    bg_loader = DataLoader(bg_dataset, batch_size=bg_batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers)

    return bg_loader, test_loader

# Generates SHAP values for all pixels in the MRIs given by the test_loader
def create_SHAP_values(bg_loader, test_loader, mri_count, save_path):
    '''
    Attributes:
        bg_loader (torch.utils.data.DataLoader): Dataloader instance for the background dataset.
        test_loader (torch.utils.data.DataLoader): Dataloader instance for the test dataset.
        mri_count (int): The total number of explanations to generate.
        save_path (str): The path to save the generated SHAP values (as .npy files).
    '''

    # while()
    batch = next(iter(bg_loader))
    bg_images, filename, _ = batch
    
    deep_explainer = shap.DeepExplainer(cnn_model, torch.unsqueeze(torch.squeeze(bg_images, 0), 1))

    iterator = iter(test_loader)
    for _ in range(mri_count):
        print(_)
        batch = next(iterator)
        test_image, filename, _ = batch
        print(filename)
        shap_values = deep_explainer.shap_values(torch.unsqueeze(test_image, 0))
        np.save(save_path+"SHAP/data/"+os.path.split(filename[0])[1].strip(".npy"), shap_values)

# Aggregates SHAP values per brain region and returns a dictionary that maps 
# each region to the average SHAP value of its pixels. 
def aggregate_SHAP_values_per_region(shap_values, seg_path, brain_regions):
    '''
    Attributes:
        shap_values (ndarray): The shap values for an MRI (.npy).
        seg_path (str): The path to the segmented MRI (.nii). 
        brain_regions (dict): The regions inside the segmented MRI image (see data_utl.py)
    '''
    region_to_avg_dict = {}

    image = nib.load(seg_path)
    image_data = image.get_fdata()

    agg_values = []

    for region in brain_regions.keys():
        agg_values.append(np.mean(shap_values, where = (region == image_data)))
        # agg_shap_value = np.mean(shap_values, where = (region == image_data))
    
    region_to_avg_dict[brain_regions[region]] = agg_values

    return region_to_avg_dict

# Returns a list containing the top-10 most contributing brain regions to each predicted class (AD/NotAD).
def output_top_10_lst(csv_file):
    '''
    Attribute:
        csv_file (str): The path to a CSV file that contains the aggregated SHAP values per region.
    '''
    # YOUR CODE HERE
    pass

# Plots SHAP values on a 2D slice of the 3D MRI. 
def plot_shap_on_mri(subject_mri, shap_values, label):
    '''
    Attributes:
        subject_mri (str): The path to the MRI (.npy).
        shap_values (str): The path to the SHAP explanation that corresponds to the MRI (.npy).
    '''
    shap_numpy = [s for s in shap_values]
    test_numpy =  np.expand_dims(subject_mri, -1)

    shap_numpy = np.squeeze(shap_numpy, 1)
    shap_numpy = np.expand_dims(shap_numpy, -1)
    shap_numpy = shap_numpy[label]

    # plot the feature attributions
    shap.image_plot(np.rot90(shap_numpy[0][91], k=1), np.rot90(test_numpy[:][:][91]),show=False)
    shap.image_plot(np.rot90(shap_numpy[0][:][109], k=1), np.rot90(test_numpy[:][:][:][109]),show=False)
    shap.image_plot(np.rot90(shap_numpy[0][:][:][91], k=1), np.rot90(test_numpy[:][:][:][:][91]),show=False)
    # shap.image_plot([np.rot90(shap_numpy[0][91], k=1), np.rot90(shap_numpy[0][91], k=1)], [np.rot90(test_numpy[:][:][91]), np.rot90(test_numpy[:][:][91])],show=False)
    plt.savefig("../output/SHAP/heatmaps/shap.png")

def write_to_csv(filepath, header, data):
    with open(filepath, 'w') as output_file:
        output_writer = csv.writer(output_file, delimiter=",")
        output_writer.writerow(header) # Write column headers
        for item in data:
            output_writer.writerow(item)


if __name__ == '__main__':

    # python explain_pipeline.py --task [1/2/3/4] --dataFolder [path to the ADNI3 folder] --outputFolder [path to the output folder where we will store the final outputs]

     # Initialize parser
    parser = argparse.ArgumentParser()

    options_list = ["task", "dataFolder", "outputFolder"]

    for argument in options_list:
        parser.add_argument("--"+argument)

    args = parser.parse_args()

    input_filepath = args.dataFolder
    output_filepath = args.outputFolder

    if(args.task == "1"):
        # TASK I: Load CNN model and instances (MRIs)
        #         Report how many of the 19 MRIs are classified correctly

        # import new CNN model
        cnn_model = _CNN(20, 0.15)

        # warm the new model with the state_dict from the checkpointed model
        checkpoint = torch.load(f=input_filepath+"/cnn_best.pth", map_location=torch.device('cpu'))
        cnn_model.load_state_dict(checkpoint.get("state_dict"))
        cnn_model.eval()

        # data loaders
        split_csv(input_filepath+"/ADNI3.csv")

        bg_csv = input_filepath+"/bg_mri_data.csv"
        test_csv = input_filepath+"/test_mri_data.csv"
        bg_loader, test_loader = prepare_dataloaders(bg_csv=bg_csv, test_csv=test_csv)

        # testing the cnn model with the test data loader
        val_output = []
        number_of_correct_predictions = 0
        with torch.no_grad():
            data_loaders = [bg_loader, test_loader]
            for loader in data_loaders:
                for image, mri_path, label in loader:
                    try:
                        output_data = cnn_model(torch.unsqueeze(torch.squeeze(image, 0), 1))
                    except:
                        output_data = cnn_model(torch.unsqueeze(image, 0))

                    for idx in range(len(output_data)):
                        data = output_data[idx]
                        curr_label = label[idx]
                        if(data[0] > data[1]):
                            val_output.append(0)
                            if(curr_label == 0):
                                number_of_correct_predictions += 1
                        else:
                            val_output.append(1)
                            if(curr_label == 1):
                                number_of_correct_predictions += 1
                
                    header = ("Classified", "Value")
                    data = [("Correct", number_of_correct_predictions), ("Incorrect", len(val_output)-number_of_correct_predictions)]
                
            # write results to CSV 
            write_to_csv(output_filepath+"/task-1.csv", header, data)


    if(args.task == "2"):
        # TASK II: Probe the CNN model to generate predictions and compute the SHAP 
        #          values for each MRI using the DeepExplainer or the GradientExplainer. 
        #          Save the generated SHAP values that correspond to instances with a
        #          correct prediction into output/SHAP/data/
        # YOUR CODE HERE
        create_SHAP_values(bg_loader, test_loader, 5, output_filepath)

    if(args.task == "3"):
        # TASK III: Plot an explanation (pixel-based SHAP heatmaps) for a random MRI. 
        #           Save heatmaps into output/SHAP/heatmaps/

        # Loading the numpy arrays
        ad_1 = np.load("../output/SHAP/data/ADNI_135_S_6510_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190823121302839_11_S863934_I1215774.npy")
        ad_1_image = np.load("../ADNI3/ADNI_135_S_6510_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190823121302839_11_S863934_I1215774.npy")

        ad_0 = np.load("../output/SHAP/data/ADNI_135_S_6446_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190711143406269_109_S840461_I1185903.npy")
        ad_0_image = np.load("../ADNI3/ADNI_135_S_6446_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190711143406269_109_S840461_I1185903.npy")
        
        # Plotting the heatmaps
        plot_shap_on_mri(ad_1_image, ad_1, 1)
        plot_shap_on_mri(ad_0_image, ad_0, 0)

    if(args.task == "4"):
        # TASK IV: Map each SHAP value to its brain region and aggregate SHAP values per region.
        #          Report the top-10 most contributing regions per class (AD/NC) as top10_{class}.csv
        #          Save CSV files into output/top10/
        # YOUR CODE HERE

        ad_0_list = ["../ADNI3/ADNI_135_S_6510_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190823121302839_11_S863934_I1215774.npy",
    "../ADNI3/ADNI_099_S_6632_MR_Accelerated_Sag_IR-FSPGR___br_raw_20200207123735297_1_S920619_I1286418.npy",
    "../ADNI3/ADNI_135_S_6446_MR_Accelerated_Sag_IR-FSPGR___br_raw_20190711143406269_109_S840461_I1185903.npy"]

        ad_1_list = ["../ADNI3/ADNI_011_S_6303_MR_Accelerated_Sagittal_MPRAGE__br_raw_20190430142811025_189_S819896_I1160021.npy",
    "../ADNI3/ADNI_022_S_6013_MR_Sagittal_3D_Accelerated_MPRAGE_br_raw_20190314145101831_129_S806245_I1142379.npy"]

        for filename in ad_0_list:
            aggregate_SHAP_values_per_region(ad_0_list, filename, brain_regions)
        
        for filename in ad_1_list:
            aggregate_SHAP_values_per_region(ad_0_list, filename, brain_regions)
