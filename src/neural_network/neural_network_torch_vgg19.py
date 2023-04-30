# total accuracy on the test data 95%

import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn

# checking if there is any GPU for faser training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# folder with data
data_dir = 'data/data_base'

# making dir for saving model
savepath = 'model'
try:
    os.mkdir(savepath)
except Exception:
    pass

# making image(data) augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# splitting data for train, test, val
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

# saving the name of the classes
class_names = image_datasets['train'].classes

# temp variables for making graphs(rating of numbers epochs for validation loss, acc; training loss, acc)
train_loss = []
val_loss = []
train_acc = []
val_acc = []
epoch_counter_train = []
epoch_counter_val = []

# making the function for training model


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                train_loss.append(running_loss/dataset_sizes[phase])
                train_acc.append(running_corrects.double() /
                                 dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "val":
                val_loss.append(running_loss / dataset_sizes[phase])
                val_acc.append(running_corrects.double() /
                               dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "val":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


model_ft = models.vgg19_bn(pretrained=True)
model_ft.classifier[6].out_features = 7

# off layers for training
for param in model_ft.features.parameters():
    param.require_grad = False

num_features = model_ft.classifier[6].in_features
features = list(model_ft.classifier.children())[:-1]
features.extend([nn.Linear(num_features, len(class_names)), nn.Softmax()])
model_ft.classifier = nn.Sequential(*features)

# making cuda render for model
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999))

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

# cleansing cuda cache
torch.cuda.empty_cache()

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=100)

# saving model
torch.save(model_ft, 'model/model_torch.pth')

# Accuracy on the test data
correct_it = 0
all_iter = 0
class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))

with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_iter += labels.size(0)
        correct_it += (predicted == labels).sum().item()

        _, predicted = torch.max(outputs, 1)
        point = (predicted == labels).squeeze()
        for j in range(len(labels)):
            label = labels[j]
            class_correct[label] += point[j].item()
            class_total[label] += 1

# printing results
print('Accuracy on the test data: %d %%' % (100 * correct_it / all_iter))

for l in range(7):
    print('Accuracy of %5s : %2d %%' % (
        class_names[l], 100 * class_correct[l] / class_total[l]))

# making dir for saving graphs
savepath = 'graphs'
try:
    os.mkdir(savepath)
except Exception:
    pass

# making graphs for presentation(val_data acc, loss)
val_acc = [i.cpu().numpy() for i in val_acc]

plt.figure()
plt.title('Ratio of number epochs for validation accuracy, loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')

plt.plot(epoch_counter_val, val_loss, color='r', label="validation loss")
plt.plot(epoch_counter_val, val_acc, color='m', label="validation accuracy")
plt.savefig('graphs/ratio_of_number_epochs_for_validation_accuracy_loss.png')
