from src.unet_parts import DoubleConv, Down, Up, OutConv
import torch.nn as nn
import cv2
import torch


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.initial = DoubleConv(n_channels, 64)
        self.conv1 = Down(64, 128)
        self.conv2 = Down(128, 256)
        self.conv3 = Down(256, 512)
        self.conv4 = Down(512, 1024)
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.final_conv = OutConv(64, 1)

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.final_conv(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    image = cv2.imread("../images/cat.png")
    image = image / 255.0
    tensor_image = torch.tensor(image).permute(2, 0, 1).float()
    tensor_image = tensor_image.unsqueeze(0)
    print(tensor_image.shape)
    model = UNet(3)
    prediction = model(tensor_image)
    print(prediction.shape)
