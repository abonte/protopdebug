import torch.nn as nn


def custom_features(pretrained=False):
    model = Custom_features()
    return model


class Custom_features(nn.Module):

    def __init__(self):
        super(Custom_features, self).__init__()
        self.kernel_sizes = []
        self.strides = []
        self.paddings = []
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            #nn.Dropout(0.2)
        )

        self.kernel_sizes.extend([3, 3, 3])
        self.strides.extend([2, 2, 1])
        self.paddings.extend([1, 1, 1])

        self.n_layers = 3

    def forward(self, x):
        return self.features(x)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network
        '''
        return self.n_layers
