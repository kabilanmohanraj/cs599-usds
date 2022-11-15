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
    batch = next(iter(bg_loader))
    bg_images, filename, labels = batch
    print("bg_image size", np.shape(bg_images)) # torch.Size([8, 182, 218, 182])
    print("\n")
    deep_explainer = shap.DeepExplainer(cnn_model, torch.unsqueeze(torch.squeeze(bg_images, 0), 1))
    print(np.shape(torch.unsqueeze(torch.squeeze(bg_images, 0), 1))) # torch.Size([8, 1, 182, 218, 182])
    print("\n\n")

    batch = next(iter(test_loader))
    test_images, filename, labels = batch
    shap_values = deep_explainer.shap_values(torch.unsqueeze(test_images, 0))
    # shap_values = deep_explainer.shap_values(torch.unsqueeze(torch.squeeze(bg_images, 0), 1))
    torch.unsqueeze(torch.squeeze(bg_images, 0), 1)

    print("test image size:", np.shape(test_images))
    print("\n")
    print(np.shape(torch.unsqueeze(test_images, 0)))
    print("\n\n")

    print(len(shap_values))
    print("\n\n")
    print("Shap value shape:" ,np.shape(shap_values))
    print("\n\n")
    print(np.sum(shap_values))
    print("\n\n")

    print(np.shape(shap_values[0][0][0][91][:][:]))

    print(np.shape(shap_values[0]))
    print(pd.DataFrame(shap_values[0][0][0][91][:][:]).shape)
    plt.imshow(shap_values[1][0][0][:][109][:])
    print(np.shape(shap_values))
    plt.savefig("./output/shap_values.png")
    plt.imshow(shap_values[1][0][0][:][:][91])
    plt.savefig("./output/shap_values1.png")
    print("\n\n")

    test = [s.shape for s in shap_values]
    print(test[0])
    print("\n\n")

    shap_numpy = [np.swapaxes(s, 1, -1) for s in shap_values]
    test_numpy = np.squeeze(np.swapaxes(test_images.numpy(), 1, -1))

    print("\n\n")
    print(np.shape(shap_numpy), np.shape(test_numpy), np.shape(test_images))
    # print(np.shape(np.squeeze(shap_numpy, 1)))
    test1 = np.squeeze(shap_numpy, 1)
    print(test1.shape)
    test1 = test1[0]
    print(test1.shape)
    print(test1[91][:][:].shape)
    print("\n\n")

    # plot the feature attributions
    shap.image_plot(shap_numpy, -test_numpy, show=False)
    # shap.image_plot(test1[91][:][:], -test_numpy, show=False)
    plt.savefig("./output/shap_values2.png")

# Aggregates SHAP values per brain region and returns a dictionary that maps 
# each region to the average SHAP value of its pixels. 
def aggregate_SHAP_values_per_region(shap_values, seg_path, brain_regions):
    '''
    Attributes:
        shap_values (ndarray): The shap values for an MRI (.npy).
        seg_path (str): The path to the segmented MRI (.nii). 
        brain_regions (dict): The regions inside the segmented MRI image (see data_utl.py)
    '''
    # YOUR CODE HERE
    pass

# Returns a list containing the top-10 most contributing brain regions to each predicted class (AD/NotAD).
def output_top_10_lst(csv_file):
    '''
    Attribute:
        csv_file (str): The path to a CSV file that contains the aggregated SHAP values per region.
    '''
    # YOUR CODE HERE
    pass

# Plots SHAP values on a 2D slice of the 3D MRI. 
def plot_shap_on_mri(subject_mri, shap_values):
    '''
    Attributes:
        subject_mri (str): The path to the MRI (.npy).
        shap_values (str): The path to the SHAP explanation that corresponds to the MRI (.npy).
    '''
    # YOUR CODE HERE
    pass

def write_to_csv(filepath, header, data):
    with open(filepath, 'w') as output_file:
        output_writer = csv.writer(output_file, delimiter=",")
        output_writer.writerow(header) # Write column headers
        for item in data:
            output_writer.writerow(item)

if __name__ == '__main__':

    output_filepath = "./output/"

    # TASK I: Load CNN model and instances (MRIs)
    #         Report how many of the 19 MRIs are classified correctly

    # import new CNN model
    cnn_model = _CNN(20, 0.15)

    # warm the new model with the state_dict from the checkpointed model
    checkpoint = torch.load(f="./ADNI3/cnn_best.pth", map_location=torch.device('cpu'))
    cnn_model.load_state_dict(checkpoint.get("state_dict"))
    cnn_model.eval()

    # data loaders
    split_csv("./ADNI3/ADNI3.csv")

    bg_csv = "./ADNI3/bg_mri_data.csv"
    test_csv = "./ADNI3/test_mri_data.csv"
    bg_loader, test_loader = prepare_dataloaders(bg_csv=bg_csv, test_csv=test_csv)

    # testing the cnn model with the test data loader
    val_output = []
    number_of_correct_predictions = 0
    with torch.no_grad():
        for test_image, test_mri_path, test_label in test_loader:
            output_data = cnn_model(torch.unsqueeze(test_image, 0))
            if(output_data[0][0] > output_data[0][1]):
                val_output.append(0)
                if(test_label == 0):
                    number_of_correct_predictions += 1
            else:
                val_output.append(1)
                if(test_label == 1):
                    number_of_correct_predictions += 1
        
            header = ("Classified", "Value")
            data = [("Correct", number_of_correct_predictions), ("Incorrect", len(val_output)-number_of_correct_predictions)]
        
        # write results to CSV 
        write_to_csv(output_filepath+"task-1.csv", header, data)


    # TASK II: Probe the CNN model to generate predictions and compute the SHAP 
    #          values for each MRI using the DeepExplainer or the GradientExplainer. 
    #          Save the generated SHAP values that correspond to instances with a
    #          correct prediction into output/SHAP/data/
    # YOUR CODE HERE
    # deep_explainer = shap.DeepExplainer()

    # * (Tensor input, Tensor weight, Tensor bias, tuple of ints stride, tuple of ints padding, tuple of ints dilation, int groups)
    #   didn't match because some of the arguments have invalid types: (DataLoader, Parameter, NoneType, tuple, tuple, tuple, int)
    # * (Tensor input, Tensor weight, Tensor bias, tuple of ints stride, str padding, tuple of ints dilation, int groups)
    #   didn't match because some of the arguments have invalid types: (DataLoader, Parameter, NoneType, tuple, tuple, tuple, int)

    create_SHAP_values(bg_loader, test_loader, 100, output_filepath)

    # TASK III: Plot an explanation (pixel-based SHAP heatmaps) for a random MRI. 
    #           Save heatmaps into output/SHAP/heatmaps/
    # YOUR CODE HERE 

    # TASK IV: Map each SHAP value to its brain region and aggregate SHAP values per region.
    #          Report the top-10 most contributing regions per class (AD/NC) as top10_{class}.csv
    #          Save CSV files into output/top10/
    # YOUR CODE HERE 


