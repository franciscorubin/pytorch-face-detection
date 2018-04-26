import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch.backends.cudnn as cudnn
import torch
import argparse
import time
import torch.nn as nn
import torchvision
from torchvision import transforms
from utils.transforms import Zoom, Translation
from utils.datasets import TransformImageDataset
from utils.utils import progress_bar
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from model import Model

class FaceDataset(Dataset):
    def __init__(self, X, y, transform=None, transform_target=None):
        self.transform = transform
        self.transform_target = transform_target
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]       
        target = self.y[idx]
        if self.transform:
            img = self.transform(img)

        if self.transform_target:
            target = self.transform_target(target)
            
        return (img, target)

checkpoint_name = 'best'
print('Starting...')

parser = argparse.ArgumentParser(description='Face Detection')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', default=False, help='choose this if you only want to test results on the current model')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
start_epoch = 0
best_acc = 0


#DATA
faces = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'data/faces_norm'), delimiter=' ')
notfaces = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'data/notfaces_norm'), delimiter=' ')
data = np.concatenate((faces, notfaces))

facesLabel = np.ones(len(faces))
notfacesLabel = np.zeros(len(notfaces))
target = np.concatenate((facesLabel, notfacesLabel))

trainX, testX, trainY, testY = train_test_split(data, target, test_size=0.2, random_state=142)
trainX = torch.FloatTensor(trainX.reshape(-1, 1, 24, 24))
testX = torch.FloatTensor(testX.reshape(-1, 1, 24, 24))
trainY = torch.FloatTensor(trainY).float().unsqueeze(1)
testY = torch.FloatTensor(testY).float().unsqueeze(1)

#transform_train = transforms.Compose([
#    transforms.ToPILImage(),
#    transforms.RandomRotation(5),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor()
#])
#
#transform_test = transforms.Compose([
#    transforms.ToTensor()
#])

#TODO add data augmentation
train_dataset = FaceDataset(trainX, trainY, transform=None) # No transforms for now
test_dataset = FaceDataset(testX, testY, transform=None) # No transforms for now

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)




checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}.ckpt'.format(checkpoint_name))
if args.resume and os.path.isfile(checkpoint_path):
   print('==> Resuming from checkpoint..')
   checkpoint = torch.load(checkpoint_path)
   net = checkpoint['net']
   best_acc = checkpoint['acc']
   start_epoch = checkpoint['epoch']
   print('Loaded best acc: {}, Starting epoch: {}'.format(best_acc, start_epoch))
else:
   print('==> Building model..')
   net = Model()

if use_cuda:
   net.cuda()
   net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
   cudnn.benchmark = True
   

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batches = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batches += 1
        train_loss += loss.data[0]
        predicted = (outputs.data >= 0.5).float()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    loss_tot = train_loss/batches
    acc = 100.*correct/total
    print('Train: Loss: {} | Acc: {} ({}/{})'.format(loss_tot, acc, correct, total))

def test(epoch, save=True):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    batches = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        batches += 1
        test_loss += loss.data[0]
        predicted = (outputs.data >= 0.5).float()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    loss_tot = test_loss/batches
    acc = 100.*correct/total
    print('Test: Loss: {} | Acc: {} ({}/{})'.format(loss_tot, acc, correct, total))

    if save:
        # Save checkpoint.
        acc = 100.*correct/total
        print('Total tested: {}, Correct tested: {}, accuracy: {}'.format(total, correct, acc))
        print('Best acc: {}'.format(best_acc))
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.module if use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_path)
            best_acc = acc


if args.test is False:
    print('==> Starting Training')
    for epoch in range(start_epoch, start_epoch+500):
        start = time.time()
        train(epoch)
        test(epoch)
        end = time.time()
        timeTaken = end - start
        print('Epoch {} took {} seconds'.format(epoch, timeTaken))
else:
    start = time.time()
    test(0, save=False)
    end = time.time()
    timeTaken = end - start
    print('Time taken: {} seconds'.format(timeTaken))


