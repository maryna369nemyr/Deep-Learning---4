"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        C,H,W = 3, 240, 240
        #num_filters = [96, 256, 384]
        num_filters = [16, 32, 64]
        weight_scale = 0.001
        K = 3 
        ####################
        #    downsampling  #
        ####################
        
        self.conv1  = nn.Conv2d(C, num_filters[0] ,kernel_size = K, stride = 1, padding = 1, bias = True)
        self.conv1.weight.data.mul_(weight_scale)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.pool1 = nn.MaxPool2d(2, stride = 2, return_indices = True)
        
        self.conv2  = nn.Conv2d(num_filters[0], num_filters[1] ,kernel_size = K, stride = 1, padding = 1, bias = True)
        self.conv2.weight.data.mul_(weight_scale)
        self.bn2 = nn.BatchNorm2d(num_filters[1])
        self.pool2 = nn.MaxPool2d(2, stride = 2, return_indices = True)
        


        self.conv3  = nn.Conv2d(num_filters[1], num_filters[2] ,kernel_size = K, stride = 1, padding = 1, bias = True)
        self.conv3.weight.data.mul_(weight_scale)     
        self.bn3 = nn.BatchNorm2d(num_filters[2])
        self.pool3 = nn.MaxPool2d(2, stride = 2, return_indices = True)
        ##############
        # upsampling #
        ##############
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)        
        self.conv_back3  = nn.Conv2d(num_filters[2], num_filters[1] ,kernel_size = K, stride = 1, padding = 1, bias = True)
        self.conv_back3.weight.data.mul_(weight_scale)     
        self.bn_back3 = nn.BatchNorm2d(num_filters[1])

        self.unpool2 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)        
        self.conv_back2  = nn.Conv2d(num_filters[1], num_filters[0] ,kernel_size = K, stride = 1, padding = 1, bias = True)
        self.conv_back2.weight.data.mul_(weight_scale)
        self.bn_back2 = nn.BatchNorm2d(num_filters[0])
               
        last = num_classes
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)        
        self.conv_back1 = nn.Conv2d(num_filters[0], last ,kernel_size = K, stride = 1, padding = 1, bias = True)
        self.conv_back1.weight.data.mul_(weight_scale)
        #self.bn_back1 = nn.BatchNorm2d(last)
        
        self.conv_tr1 = nn.ConvTranspose2d(num_filters[0], num_classes , kernel_size =5, stride=2, padding=2, output_padding=1, groups=1, bias=True)
        


        pass

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        #print("begin                      --- ", x.size())
        #print("conv1 - bn1                --- ", x.size())
        #print("conv1 - bn1 - maxpool(2,2) --- ", x.size())
        #print("conv2 - bn2 - maxpool(2,2) --- ", x.size())
        #print("upsampling conv_tr4         --- ", x.size())
        
        x, ind1 = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #print("down 1          --- ", x.size())
        x, ind2 = self.pool2(F.relu(self.bn2(self.conv2(x))))
        #print("down 2          --- ", x.size())

        x, ind3 = self.pool3(F.relu(self.bn3(self.conv3(x))))
        #print("down 3          --- ", x.size())
        
        x = F.relu(self.bn_back3(self.conv_back3(self.unpool3(x, ind3))))
        #print("up   3          --- ", x.size())

        x = F.relu(self.bn_back2(self.conv_back2(self.unpool2(x, ind2))))
        #print("up   2          --- ", x.size())
        
        #x = F.relu(self.bn_back1(self.conv_back1(self.unpool1(x, ind1))))
        #print("up   1          --- ", x.size())

        
        #x = self.conv_back1(self.unpool1(x, ind1))
        #print(" up, y -------", y.size())
        x = self.conv_tr1(x)
        #print(x.size())
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
    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
