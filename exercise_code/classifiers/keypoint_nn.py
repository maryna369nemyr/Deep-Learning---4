import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        C, H, W = 1, 96, 96
        # num_filters = [96, 256, 384]
        num_filters = [16,32,32,32]
        #num_filters = [32,64, 128, 256 ]
        weight_scale = 0.001
        k = 4

        drop = 0.1

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(C, num_filters[0], kernel_size=k, stride=1, padding=0, bias=True)
        self.conv1.weight.data.mul_(weight_scale)
        #self.bn1 = nn.BatchNorm2d(num_filters[0])
        #self.pool1 = nn.MaxPool2d(2, stride=2)



        self.dp1 = nn.Dropout2d(p = drop)
        h = H - k + 1 # conv
        h = int((h - 2)/2)  + 1 #pooling


        k=k-1
        drop = drop + 0.1
        i=0

        self.conv2 = nn.Conv2d(num_filters[i], num_filters[i+1], kernel_size=k, stride=1, padding=0, bias=True)
        self.conv2.weight.data.mul_(weight_scale)
        #self.bn2 = nn.BatchNorm2d(num_filters[1])
        self.dp2 = nn.Dropout2d(p = drop)
        h = int(( (h - k + 1) -2)/2)+1



        """
        k=k-1
        drop = drop + 0.1
        i=i+1

        self.conv3  = nn.Conv2d(num_filters[i], num_filters[i+1] ,kernel_size = k, stride = 1, padding = 0, bias = True)
        self.conv3.weight.data.mul_(weight_scale)
        #self.bn3 = nn.BatchNorm2d(num_filters[2])
        self.dp3 = nn.Dropout2d(p = drop)
        h = int(( (h - k + 1) -2)/2)+1
        
        k=k-1
        drop = drop + 0.1
        i=i+1

        self.conv4  = nn.Conv2d(num_filters[i], num_filters[i+1], kernel_size = k, stride = 1, padding = 0, bias = True)
        self.conv4.weight.data.mul_(weight_scale)
        #self.bn4 = nn.BatchNorm2d(num_filters[2])
        self.dp4 = nn.Dropout2d(p = drop)
        h = int(( (h - k + 1) -2)/2)+1
        """
        drop = drop + 0.1

        #n_in, n_out, num_classes = 6400, 1000, 2


        n_in, n_out, num_classes = 250, 100, 30

        self.fc1 = nn.Linear(num_filters [i+1] *h*h, n_out)
        self.dp5 = nn.Dropout2d(p = drop)

        drop = drop + 0.1

        self.fc2 = nn.Linear(n_out, n_out)
        self.dp6 = nn.Dropout2d(p = drop)

        self.fc3 = nn.Linear(n_out, num_classes)


        pass
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        #print("input           --- ", x.size())
        #print("conv 1          --- ", self.conv1(x).size())
        x = self.dp1(self.pool(F.relu(self.conv1(x))))
        #print("down 1          --- ", x.size())
        #print("conv 2          --- ", self.conv2(x).size())
        x = self.dp2(self.pool(F.relu(self.conv2(x))))
        #print("down 2          --- ", x.size())
        """
        x = self.dp3(self.pool(F.relu(self.conv3(x))))
        x = self.dp4(self.pool(F.relu(self.conv4(x))))
        """

        x = x.view(-1, self.num_flat_features(x))
        #print("flatten         --- ", x.size())
        x = self.dp5(F.relu(self.fc1(x)))
        #print("fc 1            --- ", x.size())
        x = self.dp6(F.relu(self.fc2(x)))
        #print("fc 2            --- ", x.size())

        x = self.fc3(x)
        #print("fc 3            --- ", x.size())



        pass
       
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        #print(x.size())
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
