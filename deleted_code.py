'''
import torch
import torch.nn as nn
import torchvision.models as models


# Load pre-trained ResNet-18 model
resnet = models.resnet18(pretrained=True)

# Modify the ResNet model for segmentation
class ResNetSegmentation(nn.Module):
    def __init__(self, original_model):
        super(ResNetSegmentation, self).__init__()
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
        )
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)  # 512 is the number of channels before the fc layer
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # Adjust scale_factor as needed

    def forward(self, x):
        x = self.features(x)  # Get feature maps
        x = self.final_conv(x)  # Apply the final convolution to output the segmentation map
        x = self.upsample(x)  # Upsample the output to match the input image size
        x = x.unsqueeze(1)  # Add channel dimension (if missing) to match target shape
        return x

# Instantiate the modified ResNet model for segmentation
resnet_segmentation = ResNetSegmentation(resnet)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_segmentation = resnet_segmentation.to(device)

# Example: Check the modified model structure
print(resnet_segmentation)







import torch.optim as optim
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange

# Example DataLoader (replace with your actual DataLoader)
train_loader = train_dl
val_loader = val_dl

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
optimizer = optim.Adam(resnet_segmentation.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in trange(num_epochs):
    resnet_segmentation.train()  # Set model to training mode
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    for i, (images, masks, _) in enumerate(tqdm(train_loader)):
        images = images.to(device).float()
        masks = masks.to(device).float()

        optimizer.zero_grad()
        outputs = resnet_segmentation(images)
        outputs = outputs.squeeze(1)  # Adjust output shape if necessary

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate per-pixel accuracy
        predicted = torch.round(torch.sigmoid(outputs))
        correct_pixels += (predicted == masks).sum().item()
        total_pixels += masks.numel()

        if i % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Step {i}/{len(train_loader)}, '
                  f'Current accuracy: {100 * correct_pixels / total_pixels:.2f}%, '
                  f'Running loss: {running_loss / (i + 1):.4f}')
            correct_pixels = 0
            total_pixels = 0

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    resnet_segmentation.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        correct_pixels = 0
        total_pixels = 0

        for images, masks, _ in tqdm(val_loader):
            images = images.to(device).float()
            masks = masks.to(device).float()
            outputs = resnet_segmentation(images)
            outputs = outputs.squeeze(1)  # Adjust output shape if necessary

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Calculate per-pixel accuracy
            predicted = torch.round(torch.sigmoid(outputs))
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += masks.numel()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, "
          f"Validation Accuracy: {100 * correct_pixels / total_pixels:.2f}%")

print("Training completed!")


'''

import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()

        base_width = 64
        width = int(out_channels * (base_width / 64.)) * 1

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        print(f"Input: {x.shape}")

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print(f"After conv1: {out.shape}")

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print(f"After conv2: {out.shape}")

        out = self.conv3(out)
        out = self.bn3(out)
        print(f"After conv3: {out.shape}")

        if self.downsample is not None:
            identity = self.downsample(x)
            print(f"Downsampled identity: {identity.shape}")

        out += identity
        out = self.relu(out)
        print(f"Output: {out.shape}")

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        print(out.shape())
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

device = 'cpu'

resnet34 = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=1).to(device)

from torchinfo import summary
print(summary(resnet34, input_shape=(4, 3, 256, 256)))