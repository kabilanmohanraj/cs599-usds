import csv
import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torch import from_numpy

# The dataset
class CNN_Data(Dataset):
    '''
        This is a custom dataset that inherits from torch.utils.data.Dataset. 
    '''
    def __init__(self, csv_dir):
        '''
        Attributes:
            csv_dir (str): The path to the CSV file that contains the MRI metadata.
        '''
        self.csv_dir = pd.read_csv(csv_dir, header=None)

    # Returns total number of data samples
    def __len__(self):
        return len(self.csv_dir)

    # Returns the actual MRI data, the MRI filename, and the label
    def __getitem__(self, idx):
        '''
        Attribute:
            idx (int): The sample MRI index.
        '''
        mri_img_path = os.path.join(self.csv_dir.iloc[idx, 0])
        mri_image = from_numpy(np.load("../data/"+mri_img_path[2:]))
        label = self.csv_dir.iloc[idx, 1]

        return mri_image, mri_img_path, label


# This is a helper function that performs the following steps:
#   1. Retrieves the metadata for the 19 MRIs provided 
#   2. Splits the 19 MRIs into two randomly selected datasets: 
#    - One that will be used for probing/testing the model (make sure it contains at least 5 MRIs).
#    - One the will be used as a background dataset for SHAP
# The function creates two new CSV files containing the metadata for each of the above datasets.
def split_csv(csv_file, output_folder='../ADNI3', random_seed = 1051):
    '''
    Attributes:
        csv_file (str): The path to the CSV file that contains the MRI metadata.
        output_folder (str): The path to store the CSV files for the test and background datasets.
        random_seed (int): The seed number to shuffle the csv_file (you can also define your own seed).
    '''
    mri_filepaths, mri_labels = read_csv(csv_file)
    mri_filePaths_in_dir = os.listdir("../data/ADNI3/")

    mri_filepaths_to_keep = []
    mri_labels_to_keep = []

    samples_in_test = 5
    
    for i in range(len(mri_filepaths)):
        if(mri_filepaths[i] in mri_filePaths_in_dir):
            mri_filepaths_to_keep.append(mri_filepaths[i])
            mri_labels_to_keep.append(mri_labels[i])
    
    # returns the path and label lists split randomly
    def randomly_partition_lists():
        # shuffling the mri filepaths list
        random.seed(random_seed)
        random.shuffle(mri_filepaths_to_keep)
        test_mri_paths = mri_filepaths_to_keep[:samples_in_test]
        bg_mri_paths = mri_filepaths_to_keep[samples_in_test:]

        # shuffling the mri labels list
        random.seed(random_seed)
        random.shuffle(mri_labels_to_keep)
        test_mri_labels = mri_labels_to_keep[:samples_in_test]
        bg_mri_labels = mri_labels_to_keep[samples_in_test:]

        return test_mri_paths, bg_mri_paths, test_mri_labels, bg_mri_labels

    test_mri_paths, bg_mri_paths, test_mri_labels, bg_mri_labels = randomly_partition_lists()

    # generate CSV files using the test and bg lists
    with open("../data"+output_folder[2:]+"/test_mri_data.csv", "w") as output_file:
        csv_writer = csv.writer(output_file, delimiter=",")
        for i in range(len(test_mri_paths)):
            csv_writer.writerow([output_folder+"/"+test_mri_paths[i], test_mri_labels[i]])
    
    with open("../data"+output_folder[2:]+"/bg_mri_data.csv", "w") as output_file:
        csv_writer = csv.writer(output_file, delimiter=",")
        for i in range(len(bg_mri_paths)):
            csv_writer.writerow([output_folder+"/"+bg_mri_paths[i], bg_mri_labels[i]])


# Returns one list containing the MRI filepaths and a second list with the respective labels
def read_csv(filename):
    '''
    Attributes:
        filename (str): The path to the CSV file that contains the MRI metadata.
    '''
    mri_filepaths = []
    mri_labels = []

    with open(filename, "r") as mri_metadata:
        csv_reader = csv.reader(mri_metadata, delimiter=",")
        csv_reader.__next__() # skip the header
        for row in csv_reader:
            mri_filepaths.append(row[1])
            mri_labels.append(row[12])
    
    return mri_filepaths, mri_labels

# Regions inside a segmented brain MRI (ONLY FOR TASK IV)
brain_regions = {1.:'TL hippocampus R',
                2.:'TL hippocampus L',
                3.:'TL amygdala R',
                4.:'TL amygdala L',
                5.:'TL anterior temporal lobe medial part R',
                6.:'TL anterior temporal lobe medial part L',
                7.:'TL anterior temporal lobe lateral part R',
                8.:'TL anterior temporal lobe lateral part L',
                9.:'TL parahippocampal and ambient gyrus R',
                10.:'TL parahippocampal and ambient gyrus L',
                11.:'TL superior temporal gyrus middle part R',
                12.:'TL superior temporal gyrus middle part L',
                13.:'TL middle and inferior temporal gyrus R',
                14.:'TL middle and inferior temporal gyrus L',
                15.:'TL fusiform gyrus R',
                16.:'TL fusiform gyrus L',
                17.:'cerebellum R',
                18.:'cerebellum L',
                19.:'brainstem excluding substantia nigra',
                20.:'insula posterior long gyrus L',
                21.:'insula posterior long gyrus R',
                22.:'OL lateral remainder occipital lobe L',
                23.:'OL lateral remainder occipital lobe R',
                24.:'CG anterior cingulate gyrus L',
                25.:'CG anterior cingulate gyrus R',
                26.:'CG posterior cingulate gyrus L',
                27.:'CG posterior cingulate gyrus R',
                28.:'FL middle frontal gyrus L',
                29.:'FL middle frontal gyrus R',
                30.:'TL posterior temporal lobe L',
                31.:'TL posterior temporal lobe R',
                32.:'PL angular gyrus L',
                33.:'PL angular gyrus R',
                34.:'caudate nucleus L',
                35.:'caudate nucleus R',
                36.:'nucleus accumbens L',
                37.:'nucleus accumbens R',
                38.:'putamen L',
                39.:'putamen R',
                40.:'thalamus L',
                41.:'thalamus R',
                42.:'pallidum L',
                43.:'pallidum R',
                44.:'corpus callosum',
                45.:'Lateral ventricle excluding temporal horn R',
                46.:'Lateral ventricle excluding temporal horn L',
                47.:'Lateral ventricle temporal horn R',
                48.:'Lateral ventricle temporal horn L',
                49.:'Third ventricle',
                50.:'FL precentral gyrus L',
                51.:'FL precentral gyrus R',
                52.:'FL straight gyrus L',
                53.:'FL straight gyrus R',
                54.:'FL anterior orbital gyrus L',
                55.:'FL anterior orbital gyrus R',
                56.:'FL inferior frontal gyrus L',
                57.:'FL inferior frontal gyrus R',
                58.:'FL superior frontal gyrus L',
                59.:'FL superior frontal gyrus R',
                60.:'PL postcentral gyrus L',
                61.:'PL postcentral gyrus R',
                62.:'PL superior parietal gyrus L',
                63.:'PL superior parietal gyrus R',
                64.:'OL lingual gyrus L',
                65.:'OL lingual gyrus R',
                66.:'OL cuneus L',
                67.:'OL cuneus R',
                68.:'FL medial orbital gyrus L',
                69.:'FL medial orbital gyrus R',
                70.:'FL lateral orbital gyrus L',
                71.:'FL lateral orbital gyrus R',
                72.:'FL posterior orbital gyrus L',
                73.:'FL posterior orbital gyrus R',
                74.:'substantia nigra L',
                75.:'substantia nigra R',
                76.:'FL subgenual frontal cortex L',
                77.:'FL subgenual frontal cortex R',
                78.:'FL subcallosal area L',
                79.:'FL subcallosal area R',
                80.:'FL pre-subgenual frontal cortex L',
                81.:'FL pre-subgenual frontal cortex R',
                82.:'TL superior temporal gyrus anterior part L',
                83.:'TL superior temporal gyrus anterior part R',
                84.:'PL supramarginal gyrus L',
                85.:'PL supramarginal gyrus R',
                86.:'insula anterior short gyrus L',
                87.:'insula anterior short gyrus R',
                88.:'insula middle short gyrus L',
                89.:'insula middle short gyrus R',
                90.:'insula posterior short gyrus L',
                91.:'insula posterior short gyrus R',
                92.:'insula anterior inferior cortex L',
                93.:'insula anterior inferior cortex R',
                94.:'insula anterior long gyrus L',
                95.:'insula anterior long gyrus R',
}