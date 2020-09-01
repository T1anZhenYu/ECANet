import torch
from torchvision import models
from torchsummary import summary
from models import resnet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet18_origin = models.resnet18().to(device)
resnet18_eca = resnet.resnet18(num_classes=1000)
summary(resnet18_origin, (3, 224, 224))
summary(resnet18_eca,(3,224,224))