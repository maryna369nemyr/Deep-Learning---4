from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
import pandas as pd
import os
import numpy as np


from exercise_code.data_utils import get_keypoints
from exercise_code.data_utils import get_image


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            custom_point (list): which points to train on
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
        self.transform = transform


    def __len__(self):
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataset                                     #
        ########################################################################

        return(self.key_pts_frame.shape[0])
        pass
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def __getitem__(self, idx):
        ########################################################################
        # TODO:                                                                #
        # Return the idx sample in the Dataset. A simple should be a dictionary#
        # where the key, value should be like                                  #
        #        {'image': image of shape [C, H, W],                           #
        #         'keypoints': keypoints of shape [num_Normalize()keypoints, 2]}          #
        # You can use mpimg.imread(image path) to read out image data          #
        ########################################################################


        #self.key_pts_frame
        map ={}

        image = get_image(idx, self.key_pts_frame)
        image = np.array(image)
        image = np.reshape(image, (1, image.shape[0], image.shape[1]))

        key_pts = get_keypoints(idx, self.key_pts_frame)


        map['image'] = image
        map['keypoints'] = np.float_(key_pts)

        t = self.transform
        if(t is not None):
            map = t(map)


        return(map)
        pass
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        if self.transform:
            sample = self.transform(sample)

        return sample


    