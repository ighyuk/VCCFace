import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import cv2
import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
import PIL.ImageOps
import glob
from efficientnet_pytorch import EfficientNet

model_name = 'efficientnet-b0'
image_size = EfficientNet.get_image_size(model_name)
# print(image_size)  # 224
model = EfficientNet.from_pretrained(model_name, num_classes=5)
# 데이터 로드!!
batch_size = 10
random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)  # 랜덤시드 고정

## make dataset
from torchvision import transforms, datasets

data_path = 'C:/projects/VCCFace/Train/'  # class 별 폴더로 나누어진걸 확 가져와서 라벨도 달아준다
gwangsu_dataset = datasets.ImageFolder(
    data_path,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
## data split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

train_idx, tmp_idx = train_test_split(list(range(len(gwangsu_dataset))), test_size=0.2, random_state=random_seed)
datasets = {}
datasets['train'] = Subset(gwangsu_dataset, train_idx)
tmp_dataset = Subset(gwangsu_dataset, tmp_idx)

val_idx, test_idx = train_test_split(list(range(len(tmp_dataset))), test_size=0.5, random_state=random_seed)
datasets['valid'] = Subset(tmp_dataset, val_idx)
datasets['test'] = Subset(tmp_dataset, test_idx)

## data loader 선언
dataloaders, batch_num = {}, {}
dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=0)
dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'],
                                                   batch_size=batch_size, shuffle=False,
                                                   num_workers=0)
dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'],
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
batch_num['train'], batch_num['valid'], batch_num['test'] = len(dataloaders['train']), len(dataloaders['valid']), len(
    dataloaders['test'])
print('batch_size : %d,  tvt : %d / %d / %d' % (batch_size, batch_num['train'], batch_num['valid'], batch_num['test']))

## 데이터 체크
import torchvision

#
# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


# num_show_img = 5

class_names = {
    "0": "Bill",
    "1": "Elon",
    "2": "Jihyo",
    "3": "Sana",
    "4": "Suzy"
}

# # train check
# inputs, classes = next(iter(dataloaders['train']))
# out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 crop
# imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
# # valid check
# inputs, classes = next(iter(dataloaders['valid']))
# out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 crop
# imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
# # test check
# inputs, classes = next(iter(dataloaders['test']))
# out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 crop
# imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])


def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) #model.state_dict은 모델의 현재 상태에 대한 참조만 반환
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0 #..?

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) #..?
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'gwangsu_predict.pt')
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(),
                         lr=0.005,
                         momentum=0.9,
                         weight_decay=1e-4)
# lr 0.005일때 predict 성능 중하
#weight_decay=1e-4 원래


lmbda = lambda epoch: 0.98739
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)
model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, criterion, optimizer_ft,
                                                                                      exp_lr_scheduler, num_epochs=50)
# torch.save(model, 'C:/Users/DeepLearning_5/PycharmProjects/FA2/Ggwangsu_predict.pt')
# image_list = glob.glob('C:/Users/DeepLearning_5/PycharmProjects/FA2/G/Test_Image/*.jpg')
# i = 0
# for i in range(len(image_list)):
#     image = cv2.imread(image_list[i], cv2.IMREAD_ANYCOLOR)
#     cv2.imshow("frame", image)
#
#     # Preprocess image
#     tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
#                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
#     img = tfms(Image.open(image_list[i])).unsqueeze(0)
#
#     # Classify
#     model.eval()
#     with torch.no_grad():
#         # print(img)
#         # imshow(img)
#         img = img.cuda()
#         outputs = model(img)
#     plt.pause(10)
#     # Print predictions
#     # torch.Tensor.cpu()
#
#     for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
#         prob = torch.softmax(outputs, dim=1)[0, idx].item()
#         print('[', class_names[str(idx)], ': {p:.2f}% ]'.format(p=prob * 100))
#     print('-----')
