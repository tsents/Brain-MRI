import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange
from sklearn.metrics import precision_score, recall_score


IMAGE_SIZE = (256, 256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def tensor_from_path(path):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    arr = cv2.resize(arr, IMAGE_SIZE)
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
        self.images = self.df.loc[:,'image_path'].values
        self.masks = self.df.loc[:,'mask_path'].values
        self.diagnosis = self.df.loc[:,'tumour'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        im_path = self.images[idx]
        msk_path= self.masks[idx]
        diagnosis = self.diagnosis[idx]
        self.im_tensor = tensor_from_path(im_path)
        self.msk_tensor= tensor_from_path(msk_path)
        return self.im_tensor.to(device), self.msk_tensor.to(device),diagnosis

def positive_negative_diagnosis(file_masks):
    mask = cv2.imread(file_masks)
    value = np.max(mask)
    if value > 0:
        return 0
    else:
        return 1

def create_data():
    # Make objects with the images and masks. 
    brain_scans = []
    mask_files = glob.glob('kaggle_3m/*/*_mask*')

    for i in mask_files:
        brain_scans.append(i.replace('_mask',''))

    # Make a dataframe with the images and their corresponding masks
    data_df = pd.DataFrame({"image_path":brain_scans, "mask_path":mask_files})

    # Apply the function to the masks and return back a column with 1 and zeros, where 0 indicate no tumor and 1 a tumor. 
    data_df["tumour"] = data_df["mask_path"].apply(lambda x: positive_negative_diagnosis(x)) 

    train_df, test_df  = train_test_split(data_df, test_size = 0.1, stratify = data_df['tumour'] )
    train_df, val_df = train_test_split(train_df,test_size = 0.2, stratify = train_df['tumour'])
    
    train_ds = data(train_df)
    test_ds = data(test_df)
    val_ds = data(val_df)

    train_dl = DataLoader(train_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=32)
    val_dl = DataLoader(val_ds, batch_size=32)

    return train_df,test_df,val_df,train_ds,test_ds,val_ds,train_dl,test_dl,val_dl

def train_model(model,train_dl,val_dl):
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 8
    for epoch in trange(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, _, labels) in enumerate(tqdm(train_dl)):
            images = images.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images).squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predicted = torch.round(outputs.data)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 40 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Step {i}/{len(train_dl)}, '
                    f'Current accuracy: {100 * correct / total:.2f}%, '
                    f'Running loss: {running_loss / (i + 1):.4f}')
                correct = 0
                total = 0

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dl):.4f}")

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, _, labels in tqdm(val_dl):
                images = images.to(device).float()
                labels = labels.to(device).float()
                outputs = model(images).squeeze()

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.round(outputs.data)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_dl):.4f}, "
            f"Validation Accuracy: {100 * val_correct / val_total:.2f}%")

    print("Training completed!")

def test_model(model,test_dl):
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad(): #In case it wasnt set on evel
        for data in test_dl:
            images, _, labels = data
            images = images.float().to(device)
            labels = labels.float().to(device)
            
            # Get model outputs
            outputs = model(images).squeeze()
            
            # Apply threshold to get binary predictions
            predicted = torch.round(outputs.data)
            
            # Flatten predicted and labels to ensure correct shapes
            predicted = predicted.view(-1)  # Flatten to 1D
            labels = labels.view(-1)        # Flatten to 1D

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total

    # Calculate precision and recall
    precision = precision_score(all_labels, all_predicted,pos_label=0)
    recall = recall_score(all_labels, all_predicted,pos_label=0)

    # Print the results
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    print('Precision of the network on the test images: %f' % precision)
    print('Recall of the network on the test images: %f' % recall)



from pytorch_grad_cam import HiResCAM, EigenCAM,AblationCAM,XGradCAM,ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

def visualize_prediction(model,target_layers,path,methods = [XGradCAM,AblationCAM]):
    input_tensor = tensor_from_path(path).float().to(device)
    input_tensor = input_tensor.unsqueeze(0)

    rgb_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.resize(rgb_img, IMAGE_SIZE)
    rgb_img = np.array(rgb_img,dtype=np.float32)
    rgb_img /= 256

    fig,ax=plt.subplots(nrows=2,ncols=2)

    predicted_class = torch.round(model(input_tensor).squeeze()).cpu().detach().numpy()
    title = 'True' if predicted_class else 'False'
    fig.suptitle(title)

    ax[0][0].axis('off')   
    ax[0][0].imshow(rgb_img)


    mask_img = cv2.imread(path.replace('.tif','_mask.tif'), cv2.IMREAD_UNCHANGED)
    mask_img = cv2.resize(mask_img, IMAGE_SIZE)

    ax[1][0].axis('off')   
    ax[1][0].imshow(mask_img,cmap='gray')

    i = 2
    for method in methods:
        cam = method(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True,image_weight=0.5)

        ax[i%2][i//2].axis('off')
        ax[i%2][i//2].imshow(visualization)
        i+=1
        plt.plot()


class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        # Averaging the outputs from both models
        x = (x1 + x2) / 2
        return x

def train_ensamble(ensemble_model,train_dl,val_dl):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(ensemble_model.parameters(), lr=0.001)


    num_epochs = 8

    for epoch in trange(num_epochs):
        ensemble_model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, _, labels) in enumerate(tqdm(train_dl)):
            images = images.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = ensemble_model(images).squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predicted = torch.round(outputs.data)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 40 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Step {i}/{len(train_dl)}, '
                    f'Current accuracy: {100 * correct / total:.2f}%, '
                    f'Running loss: {running_loss / (i + 1):.4f}')
                correct = 0
                total = 0

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dl):.4f}")

    # Validation
    ensemble_model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    for images, _, labels in tqdm(val_dl):
        images = images.to(device).float()
        labels = labels.to(device).float()
        outputs = ensemble_model(images).squeeze()

        loss = criterion(outputs, labels)
        val_loss += loss.item()

        predicted = torch.round(outputs.data)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

    print(f"Validation Loss: {val_loss / len(val_dl):.4f}, "
          f"Validation Accuracy: {100 * val_correct / val_total:.2f}%")

    print("Training completed!")

# Grad-CAM visualization function
def visualize_ensemble_gradcam(path, ensamble, target_layers_a,target_layers_b):
    input_tensor = tensor_from_path(path).float().to(device)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Load and prepare the original image
    rgb_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.resize(rgb_img, IMAGE_SIZE)
    rgb_img = np.array(rgb_img, dtype=np.float32)
    rgb_img /= 255.0  # Normalize to [0, 1]

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle("Combined Grad-CAM from ResNet and EfficientNet")

    # Display the original image
    ax[0][0].axis('off')
    ax[0][0].imshow(rgb_img)

    # Load and display the corresponding mask image (if available)
    mask_img = cv2.imread(path.replace('.tif', '_mask.tif'), cv2.IMREAD_UNCHANGED)
    if mask_img is not None:
        mask_img = cv2.resize(mask_img,IMAGE_SIZE)
        ax[1][0].axis('off')
        ax[1][0].imshow(mask_img, cmap='gray')

    # Apply Grad-CAM for ResNet
    cam_resnet = EigenCAM(model=ensamble, target_layers=target_layers_a)
    grayscale_cam_resnet = cam_resnet(input_tensor=input_tensor)[0]

    # Apply Grad-CAM for EfficientNet
    cam_efficientnet = EigenCAM(model=ensamble, target_layers=target_layers_b)
    grayscale_cam_efficientnet = cam_efficientnet(input_tensor=input_tensor)[0]

    # Optionally display individual CAMs
    visualization_resnet = show_cam_on_image(rgb_img, grayscale_cam_resnet, use_rgb=True)
    visualization_efficientnet = show_cam_on_image(rgb_img, grayscale_cam_efficientnet, use_rgb=True)

    ax[0][1].axis('off')
    ax[0][1].imshow(visualization_resnet)
    ax[1][1].axis('off')
    ax[1][1].imshow(visualization_efficientnet) 

    plt.show()


label_classes = ['RNASeqCluster','MethylationCluster','miRNACluster','CNCluster','RPPACluster','OncosignCluster','COCCluster','histological_type','neoplasm_histologic_grade']

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, data_frame,use_device = device):
        self.root_dir = root_dir
        self.labels = data_frame
        self.labels.dropna(subset=['tumor_location'], inplace=True)
        encoder = OneHotEncoder(sparse_output=False)
        self.rna = encoder.fit_transform(self.labels[label_classes])
        self.transform = ToTensor()
        self.use_device = use_device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels.loc[idx, 'death01']
        folder_name = self.labels.iloc[idx, 0]

        images = []

        for dir_name in os.listdir(self.root_dir):
            if dir_name.startswith(folder_name):
                img_folder = os.path.join(self.root_dir, dir_name)

                for img_name in sorted(os.listdir(img_folder)):
                    if not img_name.endswith('mask.tif'):
                        img_path = os.path.join(img_folder, img_name)

                        image = Image.open(img_path)
                        image = self.transform(image)
                        images.append(image)

        images = torch.stack(images)

        RNA_dat = self.rna[idx]
        return images.float().to(self.use_device), torch.tensor([label]).float().to(self.use_device), torch.tensor(RNA_dat).float().to(self.use_device)

from sklearn.preprocessing import OneHotEncoder
from torchvision.transforms import ToTensor

def death_data(use_device=device):
    df = pd.read_csv('kaggle_3m/data.csv')
    df.dropna(subset=['death01'], inplace=True)
    df = df.fillna(0)
    # df = df.replace({'death01' : class_mapping})

    dataset = ImageFolderDataset(root_dir='kaggle_3m', data_frame=df,use_device=use_device)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    return train_loader,test_loader
    

def death_train(model,train_loader):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01,amsgrad=True)
    num_epochs = 15
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (inputs, labels,rna) in enumerate(train_loader):

            outputs = model(inputs,rna)
            loss = criterion(outputs, labels)

            predicted = torch.round(outputs.data)
            total += 1
            # print(predicted,labels)
            correct += (predicted == labels).sum().item()
            loss.backward()

            if i % 4 == 0: ## update the model in batches, hopefully more stable
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.empty_cache()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}%')

def death_test(model, test_loader):
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad(): # In case it wasn't set on eval
        for data in test_loader:
            inputs, labels, rna = data
            outputs = model(inputs, rna)


            # Apply threshold to get binary predictions
            predicted = torch.round(outputs.data)
            
            predicted = predicted.view(-1)
            labels = labels.view(-1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # # Calculate accuracy
    accuracy = 100 * correct / total

    # Calculate precision and recall
    precision = precision_score(all_labels, all_predicted, pos_label=0)
    recall = recall_score(all_labels, all_predicted, pos_label=0)

    # Print the results
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    print('Precision of the network on the test images: %f' % precision)
    print('Recall of the network on the test images: %f' % recall)