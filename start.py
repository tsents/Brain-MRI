import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader

PATH = "kaggle_3m"
IMAGE_SIZE = (256, 256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

paths = []

for dirname in os.listdir(PATH):
    if os.path.isdir(os.path.join(PATH, dirname)):
        
        for filename in os.listdir(os.path.join(PATH, dirname)):
            if filename.endswith('.tif'):
                paths.append(PATH+'/'+dirname+'/'+filename)


def data_frame(data):
    images = list(filter(lambda x: not x.endswith('mask.tif'), data))
    images.sort(key=lambda x: int(x.rsplit('_', 3)[-1][:-4]))
    images.sort(key=lambda x: int(x.rsplit('_', 3)[-2]))
    
    IDs = list(map(lambda x: x.rsplit('/', 3)[-1][:-4], images))

    masks = list(filter(lambda x: x.endswith('mask.tif'), data))
    masks.sort(key=lambda x: int(x.rsplit('_', 3)[-2]))
    masks.sort(key=lambda x: int(x.rsplit('_', 3)[-3]))

    pixels = lambda x: Image.open(x)
    largest_pixel = lambda y: np.max(pixels(y))
    diagnotic_function = lambda z: 1 if largest_pixel(z) > 0 else 0
    diagnoses = list(map(lambda x: diagnotic_function(x), masks))

    data_df = pd.DataFrame({'ID': IDs, 'Image': images, 'Mask': masks, 'Diagnosis': diagnoses})
    
    train_index, val_index = train_test_split(data_df.index.values.tolist(), test_size=0.19, random_state=42)
    val_index, test_index = train_test_split(val_index, test_size=0.12, random_state=42)
    
    # Making train, test, and validation dataframes
    train_df, val_df, test_df = data_df.iloc[train_index], data_df.iloc[val_index], data_df.iloc[test_index]
    
    return train_df, val_df, test_df
    
# Making the dataframes
train_df, val_df, test_df = data_frame(paths)

print(len(train_df), len(val_df), len(test_df))

print(train_df.head())

def tensor_from_path(path):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    arr = cv2.resize(arr, (224,224))
    arr = arr / 255
    if len(arr.shape) == 3:
        tensor = torch.tensor(arr).permute(2,0,1)
    elif len(arr.shape) == 2:
        tensor = torch.tensor(arr).unsqueeze(0)
    else:
        raise ValueError(f"Expected data shape to be (..., ..., 3) or (..., ...) , but got {arr.shape}")
    return tensor

class data(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df        
        self.images = self.df.loc[:,'Image'].values
        self.masks = self.df.loc[:,'Mask'].values
        self.diagnosis = self.df.loc[:,'Diagnosis'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        im_path = self.images[idx]
        msk_path= self.masks[idx]
        diagnosis = self.diagnosis[idx]
        self.im_tensor = tensor_from_path(im_path)
        self.msk_tensor= tensor_from_path(msk_path)
        return self.im_tensor.to(device), self.msk_tensor.to(device),diagnosis

train_ds = data(train_df)
test_ds = data(test_df)

train_dl = DataLoader(train_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)  # Adjusted based on the actual size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

net = Net().to(device)

import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

dataloader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)

for epoch in range(2):
    correct = 0
    total = 0
    current_predicted = 0
    current_total = 0
    for i, data in enumerate(dataloader, 0):

        inputs, _,labels = data
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        outputs = net(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = torch.round(outputs.data) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        current_total += labels.size(0)
        current_predicted += (predicted == labels).sum().item()
        if i % 100 == 0:
            print('Current accaracy: %d %%' % (100 * current_predicted / current_total))
            print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
            current_predicted = 0
            current_total = 0


print('Finished Training')

test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=True)

with torch.no_grad():  # we don't need gradients for test, so it will save memory
    correct = 0
    total = 0
    for data in test_dataloader:
        images,_, labels = data
        images = images.float().to(device)
        labels = labels.float().to(device)
        outputs = net(images).squeeze()
        predicted = torch.round(outputs.data)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(correct,total,np.shape(predicted),np.shape(labels))
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))










import torch
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Define the function to load and preprocess the image
def get_img_array(img_path, size):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    return img

# Function to generate the Grad-CAM heatmap
def make_gradcam_heatmap(img_tensor, model, target_layer):
    model.eval()
    img_tensor = img_tensor.to(device)
    
    # Forward pass to get predictions and features from the target layer
    features = None
    def hook_function(module, input, output):
        nonlocal features
        features = output
    
    handle = target_layer.register_forward_hook(hook_function)
    
    outputs = model(img_tensor)
    handle.remove()
    
    # Get the index of the predicted class
    pred_index = outputs.argmax(dim=1).item()
    class_output = outputs[:, pred_index]
    
    # Backward pass to get gradients of the target layer
    model.zero_grad()
    class_output.backward(retain_graph=True)
    
    gradients = target_layer.weight.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Multiply each channel in the feature map by the corresponding gradient
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]
    
    # Average the channels of the feature map
    heatmap = features.mean(dim=1).squeeze()
    
    # Normalize the heatmap between 0 and 1
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap /= heatmap.max()
    
    return heatmap

# Define the path to the image and the target layer
img_path = "path_to_your_image.jpg"
target_layer = net.conv2

# Load and preprocess the image
img_size = (256, 256)
img_tensor = get_img_array(img_path, img_size)

# Generate the Grad-CAM heatmap
heatmap = make_gradcam_heatmap(img_tensor, net, target_layer)

# Display the heatmap
plt.matshow(heatmap)
plt.show()

# Superimpose the heatmap on the original image
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

# Display the superimposed image
cv2.imshow('Grad-CAM', superimposed_img / 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
