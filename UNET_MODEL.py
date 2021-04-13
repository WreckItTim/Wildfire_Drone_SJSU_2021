# using padded convs to avoid cropping and stitching images
import torch #pip install pytorch (in anaconda)
import torch.nn as nn
import torchvision.transforms.functional as TF # will be installed with toch


'''
Tim: I need this to load the model
'''

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        return self.conv(X)


class UNET(nn.Module):
    # binary classes
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        # store the conv layers
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # right side, conv layers
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # left: Transpose convs
        for feature in reversed(features):
            # upsampling
            self.ups.append(
                nn.ConvTranspose2d( feature*2, feature, kernel_size=2, stride=2)
            )
            # 2 convs
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # final 1x1 conv: changes the number of channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # reverse list
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # tanspose 2d
            x = self.ups[idx](x)
            # get skip connections
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                # add padding: this is useful since after the max pool images will always be smaller
                x = TF.resize(x, size=skip_connection.shape[2:])

            # add skip in channels
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# def test():
#     x = torch.randn((3,1,128, 128))
#     model = UNET(in_channels=1, out_channels=1)
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape == x.shape
#
#
# if __name__ == "__main__":
#     test()
