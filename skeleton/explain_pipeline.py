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

from PIL import Image

from model import _CNN
from data_util import split_csv
from data_util import CNN_Data
from data_util import brain_regions

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
    all_bg_images = []

    iterator = iter(bg_loader)
    for _, sample in enumerate(bg_loader):
        bg_images, filename, _ = sample
        all_bg_images += bg_images
    
    # initialize the DeepExplainer model
    deep_explainer = shap.DeepExplainer(cnn_model, torch.unsqueeze(torch.squeeze(bg_images, 0), 1))

    iterator = iter(test_loader)
    for _ in range(mri_count):
        batch = next(iterator)
        test_image, filename, _ = batch
        shap_values = deep_explainer.shap_values(torch.unsqueeze(test_image, 0))
        np.save(save_path+"/SHAP/data/"+os.path.split(filename[0])[1].strip(".npy"), shap_values)

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

    for region in brain_regions.keys():
        sum_shap_value = np.sum(shap_values, where = (region == image_data))
        pixel_count = np.count_nonzero(region == image_data)
        region_to_avg_dict[region] = [sum_shap_value, pixel_count]

    return region_to_avg_dict

# Returns a list containing the top-10 most contributing brain regions to each predicted class (AD/NotAD).
def output_top_10_lst(csv_file):
    '''
    Attribute:
        csv_file (str): The path to a CSV file that contains the aggregated SHAP values per region.
    '''
    data_to_sort = []

    with open(csv_file) as input_file:
        input_reader = csv.reader(input_file, delimiter=",")
        next(input_reader)
        for row in input_reader:
            data_to_sort.append(row)
    sorted_list = sorted(data_to_sort, key=lambda item: item[2], reverse=True)
    
    return sorted_list[:10]

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

    # def combine_shap_plots():
    #     slice_1 = Image.open(output_filepath+"/SHAP/heatmaps/shap-slice-1.png")
    #     slice_2 = plt.imread(output_filepath+"/SHAP/heatmaps/shap-slice-2.png")
    #     slice_3 = plt.imread(output_filepath+"/SHAP/heatmaps/shap-slice-3.png")

    #     slice_1.size
    #     slice_2.size
    #     slice_3.size

    #     slice_1 = slice_1.resize((250, 90))
    #     slice_2 = slice_2.resize((250, 90))
    #     slice_3 = slice_3.resize((250, 90))

    #     image_combined = Image.new("RGB", (600, 1500), "white")
    #     image_combined.paste(slice_1, (0,0))
    #     image_combined.paste(slice_2, (0,500))
    #     image_combined.paste(slice_3, (0,1000))

    #     plt.savefig(output_filepath+"/SHAP/heatmaps/shap-combined-"+str(label)+".png")


    # print(np.shape(shap_numpy[0][:, :, 91]), np.shape(test_numpy[:, :, 91]))

    # shap_value_slices = np.array([[shap_numpy[0, :, :, 91]], [shap_numpy[0, :, 109, :]], [shap_numpy[0, 91, :, :]]])
    # test_numpy_slices = np.array([test_numpy[:, :, 91], test_numpy[:, 109, :], test_numpy[91, :, :]])
    # print(np.shape(shap_value_slices))

    # shap_image_pair = [shap_numpy[0][:, :, 91], test_numpy[:, :, 91]]
    # shap_image_pair = np.array(shap_image_pair)
    # shap_image_pair = np.expand_dims(shap_image_pair, -1)
    # print(np.shape(shap_image_pair))
    # shap.image_plot(shap_value_slices, test_numpy_slices, show=False)

    # plot the feature attributions

    ## (NOTE) One image for each 2D slice in the 3D MRI image
    shap.image_plot(np.rot90(shap_numpy[0][:, :, 91], k=1), np.rot90(test_numpy[:, :, 91]),show=False)
    plt.savefig(output_filepath+"/SHAP/heatmaps/"+str(label)+"-shap-slice-1.png")
    

    shap.image_plot(np.rot90(shap_numpy[0][:, 109, :], k=1), np.rot90(test_numpy[:, 109, :]),show=False)
    plt.savefig(output_filepath+"/SHAP/heatmaps/"+str(label)+"-shap-slice-2.png")
    

    shap.image_plot(np.rot90(shap_numpy[0][91, :, :], k=1), np.rot90(test_numpy[91, :, :]),show=False)
    plt.savefig(output_filepath+"/SHAP/heatmaps/"+str(label)+"-shap-slice-3.png")

    # combine_shap_plots()


# Utility method to write data to csv file
def write_to_csv(filepath, header, data):
    with open(filepath, 'w') as output_file:
        output_writer = csv.writer(output_file, delimiter=",")
        output_writer.writerow(header) # Write column headers
        if(isinstance(data, dict)):
            for item in data.items():
                output_writer.writerow([item[0], item[1][1], item[1][0]])
        else:
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

    if(input_filepath[-1] == os.sep):
        input_filepath = input_filepath[:-1]
    
    if(output_filepath[-1] == os.sep):
        output_filepath = output_filepath[:-1]

    
    # TASK I: Load CNN model and instances (MRIs)
    #         Report how many of the 19 MRIs are classified correctly

    # data loaders
    split_csv(input_filepath+"/ADNI3.csv")

    bg_csv = input_filepath+"/bg_mri_data.csv"
    test_csv = input_filepath+"/test_mri_data.csv"
    bg_loader, test_loader = prepare_dataloaders(bg_csv=bg_csv, test_csv=test_csv)

    # import new CNN model
    cnn_model = _CNN(20, 0.15)

    # warm the new model with the state_dict from the checkpointed model
    checkpoint = torch.load(f=input_filepath+"/cnn_best.pth", map_location=torch.device('cpu'))
    cnn_model.load_state_dict(checkpoint.get("state_dict"))
    cnn_model.eval()

    if(args.task == "1"):
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

        ad_0_segpath = []
        ad_1_segpath = []

        ad_0_shap_values = []
        ad_1_shap_values = []

        with open(input_filepath+"/test_mri_data.csv") as test_mri_csv:
            data = test_mri_csv.readlines()
            for line in data:
                if(line[-2] == "0"):
                    ad_0_shap_values.append(input_filepath+"/"+os.path.split(line.split(",")[0])[1])
                    temp_segpath = os.path.split(line.split(",")[0])[1].strip(".npy") + ".nii"
                    ad_0_segpath.append(input_filepath+"/seg/"+temp_segpath)
                if(line[-2] == "1"):
                    ad_1_shap_values.append(input_filepath+"/"+os.path.split(line.split(",")[0])[1])
                    temp_segpath = os.path.split(line.split(",")[0])[1].strip(".npy") + ".nii"
                    ad_1_segpath.append(input_filepath+"/seg/"+temp_segpath)

        ad_0_dict = {} # to store sum and count for aggregation
        ad_1_dict = {} # to store sum and count for aggregation
        temp = {}
        header = ("Region number", "region", "value")

        for i in range(len(ad_0_segpath)):
            temp = aggregate_SHAP_values_per_region(np.load(ad_0_shap_values[i]), ad_0_segpath[i], brain_regions)

            if(len(ad_0_dict) == 0):
                ad_0_dict = temp
            for key in brain_regions.keys():
                ad_0_dict[key] = [ad_0_dict[key][0] + temp[key][0], ad_0_dict[key][1] + temp[key][1]]
            
            if(i == len(ad_0_segpath)-1):
                for key in brain_regions.keys():
                    ad_0_dict[key] = [ad_0_dict[key][0] / ad_0_dict[key][1], brain_regions[key]]
        
         # writing all aggregated shap values to CSV
        write_to_csv(output_filepath+"/agg_ad_0.csv", header, ad_0_dict)

        # sorting the list based on the average shap values
        sorted_data = output_top_10_lst(output_filepath+"/agg_ad_0.csv")

        # writing to output file
        write_to_csv(output_filepath+"/task-4-false.csv", header, sorted_data)

        for i in range(len(ad_1_segpath)):
            temp = aggregate_SHAP_values_per_region(np.load(ad_1_shap_values[i]), ad_1_segpath[i], brain_regions)

            if(len(ad_1_dict) == 0):
                ad_1_dict = temp
            for key in brain_regions.keys():
                ad_1_dict[key] = [ad_1_dict[key][0] + temp[key][0], ad_1_dict[key][1] + temp[key][1]]
            
            if(i == len(ad_1_segpath)-1):
                for key in brain_regions.keys():
                    ad_1_dict[key] = [ad_1_dict[key][0] / ad_1_dict[key][1], brain_regions[key]]
        
        # writing all aggregated shap values to CSV
        write_to_csv(output_filepath+"/agg_ad_1.csv", header, ad_1_dict)

        # sorting the list based on the average shap values
        sorted_data = output_top_10_lst(output_filepath+"/agg_ad_1.csv")

        # writing to output file
        write_to_csv(output_filepath+"/task-4-true.csv", header, sorted_data)
