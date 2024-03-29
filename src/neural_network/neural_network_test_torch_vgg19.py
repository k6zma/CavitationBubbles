import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from sklearn.metrics import classification_report

# checking for available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data augmentation
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

# here u need to write the directory with yout data
data_dir = 'data/data_base'

# make datasets from our images
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
print('image_datasets loaded')
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val', 'test']}
print('dataloaders loaded')
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
print(dataset_sizes)

# saving the name of the classes
class_names = image_datasets['train'].classes
print(class_names)

# here u need to write the path with the neural_network model
model_ft = torch.load(
    'data/model/model_torch/model_torch_vgg19.pth', map_location=device)

# total accuracy on the test data_base
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in tqdm(enumerate(dataloaders['test'])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

# total accuracy on the test data_base of each classes
class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))
with torch.no_grad():
    for i, (inputs, labels) in tqdm(enumerate(dataloaders['test'])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, predicted = torch.max(outputs, 1)
        point = (predicted == labels).squeeze()
        for j in range(len(labels)):
            label = labels[j]
            class_correct[label] += point[j].item()
            class_total[label] += 1

for i in range(7):
    print('Accuracy of %5s : %2d %%' % (
        class_names[i], 100 * class_correct[i] / class_total[i]))

# classification report
test_labels = []
test_preds = []

with torch.no_grad():
    for i, (inputs, labels) in tqdm(enumerate(dataloaders['test'])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_labels += list(labels.cpu().numpy())
        test_preds += list(predicted.cpu().numpy())

print(classification_report(test_labels, test_preds, target_names=class_names))
