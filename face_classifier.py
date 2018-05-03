import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch.backends.cudnn as cudnn
import time
import numpy as np
import torch
import config
from model import Model
import torchvision
from torchvision import transforms

classifier_checkpoint_name = 'with_celebA_data'
classifier_checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}.ckpt'.format(classifier_checkpoint_name))
checkpoint = torch.load(classifier_checkpoint_path, map_location=lambda storage, loc: storage)

net = checkpoint['net']
if torch.cuda.is_available():
  net = net.cuda()
  net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
  cudnn.benchmark = True

# TODO imsize and transform_test are copied from face_classifier_training. Import them or merge this two files (isFace method also contains stuff of the test method of training)
imsize = (24, 24)

transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(imsize),
    transforms.ToTensor()
])


# Input: normalized 24x24 image
def isFace(image):
  img = transform_test(image)
  if torch.cuda.is_available():
    img = img.cuda()
  img = img.view(-1, 1, 24, 24)
  img = torch.autograd.Variable(img, volatile=True)
  
  prediction = net(img)
  return prediction.data[0][0] >= config.THRESHOLD


if __name__ == '__main__':
  faces = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'data/faces_norm'), delimiter=' ')
  notfaces = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'data/notfaces_norm'), delimiter=' ')

  imageFace = faces[0]
  imageNotFace = notfaces[0]

  start = time.time()
  pred = isFace(imageFace)
  end = time.time()
  timeTaken = end - start
  print('Face predicted value: {}, time taken: {} seconds'.format(pred, timeTaken))

  start = time.time()
  pred = isFace(imageNotFace)
  end = time.time()
  timeTaken = end - start
  print('NotFace predicted value: {}, time taken: {} seconds'.format(pred, timeTaken))