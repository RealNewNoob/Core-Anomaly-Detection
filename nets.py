import torch
import torch.nn as nn
import copy
# Convolutional autoencoder with 5 convolutional blocks (BN version)
class Core_CAE(nn.Module):
    def __init__(self, input_shape=[128,128,1], filters=[32, 64, 128, 256, 512], leaky=True, neg_slope=0.01, 
                 center_neurons = 100, sig_activations=False, bias=True):
        super(Core_CAE, self).__init__()
        self.sig_activations = sig_activations
        self.pretrained = False
        self.input_shape = input_shape
        self.filters = filters
        self.relu = nn.ReLU(inplace=False)
        self.center_neurons = center_neurons
        
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias) 
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.bn3_1 = nn.BatchNorm2d(filters[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], 5, stride=2, padding=2, bias=bias)
        self.bn4_1 = nn.BatchNorm2d(filters[3])
        self.conv5 = nn.Conv2d(filters[3], filters[4], 3, stride=2, padding=1, bias=bias)

        lin_features_len = ((input_shape[0] // 2 // 2 // 2 // 2) // 2) * (
                    (input_shape[0] // 2 // 2 // 2 // 2) // 2) * filters[4]
        self.embedding = nn.Linear(lin_features_len, center_neurons, bias=bias)
        self.deembedding = nn.Linear(center_neurons, lin_features_len, bias=bias)
        
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv5 = nn.ConvTranspose2d(filters[4], filters[3], 3, stride=2, padding=1, output_padding=out_pad,
                                          bias=bias)
        self.bn5_2 = nn.BatchNorm2d(filters[3])
        
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn4_2 = nn.BatchNorm2d(filters[2])
        
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.bn4_1(x)
        x = self.conv5(x)
        if self.sig_activations:
            x = self.sig(x)
        else:
            x = self.relu5_1(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.deembedding(x)
        x = self.relu5_2(x)
        x = x.view(x.size(0), self.filters[4], ((self.input_shape[0]//2//2//2//2) // 2), ((self.input_shape[0]//2//2//2//2) // 2))
        x = self.deconv5(x)
        x = self.relu4_2(x)
        x = self.bn5_2(x)
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        if self.sig_activations:
            x = self.tanh(x)
        return x