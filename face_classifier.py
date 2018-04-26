import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch.backends.cudnn as cudnn
import time
import numpy as np
import torch
from model import Model

classifier_checkpoint_name = 'best_full_image'
classifier_checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}.ckpt'.format(classifier_checkpoint_name))
checkpoint = torch.load(classifier_checkpoint_path, map_location=lambda storage, loc: storage)

net = checkpoint['net']
if torch.cuda.is_available():
  net = net.cuda()
  net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
  cudnn.benchmark = True

# Input: normalized 24x24 image
def isFace(image):
  img = image.reshape(1, 1, 24, 24)
  inp = torch.autograd.Variable(torch.FloatTensor(img), volatile=True)
  if torch.cuda.is_available():
    inp = inp.cuda()
  prediction = net(inp)
  return prediction.data[0][0] >= 0.9


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