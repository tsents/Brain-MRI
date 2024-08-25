import res_define

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
# device = res_define.device
device = 'cpu'

class CombinedModel(nn.Module):
    def __init__(self,img_model,img_size,fc1_size,fc2_size):
        super(CombinedModel, self).__init__()
        self.img_model = img_model
        self.fc1 = nn.Linear(36 + img_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)

    def forward(self, x,rna):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(timesteps, C, H, W)
        x = self.img_model(x)
        x = x.view(batch_size, timesteps, -1)
        x = torch.mean(x,1)
        x = torch.cat((x,rna),1)
        x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
resnet3 = models.resnet18(pretrained=True)

embedding_size = 5

resnet3.fc = nn.Sequential(
    nn.Linear(resnet3.fc.in_features, embedding_size),
)

resnet3 = resnet3.to(device)
deathnet = CombinedModel(resnet3,embedding_size,50,30).to(device)

death_train,death_test = res_define.death_data('cpu')
res_define.death_train(deathnet,death_train)
torch.save(deathnet.state_dict(), 'deathnet.pth')
