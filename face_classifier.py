import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch.backends.cudnn as cudnn
import time
import numpy as np
import torch
import config
import glob
from model import Model
import torchvision
from torchvision import transforms
from PIL import Image
import time
import random
import math

net = None
threshold = 0.999
classifier_checkpoint_name = 'model4_FULL_augmented'

def init(checkpoint_name=None, threshold_=None):
    global classifier_checkpoint_name
    global threshold
    global net

    if checkpoint_name:
        classifier_checkpoint_name = checkpoint_name

    if threshold_:
        threshold = threshold_

    print('Classifying with model: {}'.format(classifier_checkpoint_name))
    classifier_checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint/{}.ckpt'.format(classifier_checkpoint_name))
    checkpoint = torch.load(classifier_checkpoint_path, map_location=lambda storage, loc: storage)

    net = checkpoint['net']
    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    return checkpoint

# TODO imsize and transform_test are copied from face_classifier_training. Import them or merge this two files (isFace method also contains stuff of the test method of training)
imsize = (24, 24)

transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(imsize),
    transforms.ToTensor()
])

transform_test_no_greyscale = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

# Input: pil image
def isFace(image, already_greyscale=False):
  if already_greyscale:
      img = transform_test_no_greyscale(image)
  else:
      img = transform_test(image)

  if torch.cuda.is_available():
      img = img.cuda()

  img = img.view(-1, 1, 24, 24)

  with torch.no_grad():
      prediction = net(img)
  return prediction.data[0][0] >= threshold

batch_size = 1024
def getFaces(imags, already_greyscale=False):
    startTime = time.time()
    if already_greyscale:
        img_list = torch.stack([transform_test_no_greyscale(image) for image in imags])
    else:
        img_list = torch.stack([transform_test(image) for image in imags])

    transformEndTime = time.time()

    if torch.cuda.is_available():
        img_list = img_list.cuda()

    img_list = img_list.view(-1, 1, 24, 24)
    l = len(img_list)
    n = math.ceil(l / batch_size)
    res = []
    with torch.no_grad():
        for i in range(0, n):
            prediction = net(img_list[i*batch_size:(i*batch_size) + batch_size])
            if torch.cuda.is_available():
                prediction = prediction.cpu()
            pred_res = np.squeeze((prediction >= threshold).numpy().astype('bool'))
            res.extend(pred_res)
    return res

if __name__ == '__main__':
    init()
    
    faces = glob.glob('{}/*'.format(os.path.join(os.path.dirname(__file__), 'data/extracted_faces')))
    nofaces = glob.glob('{}/*'.format(os.path.join(os.path.dirname(__file__), 'data/extracted_nofaces')))

    timeTakenFace = []
    timeTakenNoFace = []
    for i in range(0, 1000):
        imageFace = Image.open(faces[random.randint(0, len(faces))]).convert('L')
        imageNotFace = Image.open(nofaces[random.randint(0, len(nofaces))]).convert('L')

        start = time.time()
        pred = isFace(imageFace, already_greyscale=True)
        end = time.time()
        timeTaken = end - start
#        print('Face predicted value: {}, time taken: {} seconds'.format(pred, timeTaken))
        timeTakenFace.append(timeTaken)

        start = time.time()
        pred = isFace(imageNotFace, already_greyscale=True)
        end = time.time()
        timeTaken = end - start
#        print('NotFace predicted value: {}, time taken: {} seconds'.format(pred, timeTaken))
        timeTakenNoFace.append(timeTaken)

    print('Avg Time Taken For Faces: {} seconds'.format(np.array(timeTakenFace).mean()))
    print('Avg Time Taken For NotFaces: {} seconds'.format(np.array(timeTakenNoFace).mean()))