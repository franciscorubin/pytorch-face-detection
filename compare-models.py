import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from PIL import Image
import glob
from face_classifier import getFaces, init
import random 
from pytorch_utils.helpers import Metrics
import numpy as np
import time 

random.seed(213123)
metrics = Metrics(os.path.join(os.path.dirname(__file__), 'logs/comparison.json'))

def evaluate(checkpoint_name):
    init(checkpoint_name=checkpoint_name)

    faces = glob.glob('{}/*'.format(os.path.join(os.path.dirname(__file__), 'data/extracted_faces')))
    nofaces = glob.glob('{}/*'.format(os.path.join(os.path.dirname(__file__), 'data/extracted_nofaces')))

    facesSet = faces[random.sample(range(0, len(faces)), 512)]
    nofacesSet = nofaces[random.sample(range(0, len(nofaces)), 512)]
    X = np.concatenate((facesSet, nofacesSet))

    facesLabel = np.ones(len(facesSet))
    notfacesLabel = np.zeros(len(nofacesSet))
    y = np.concatenate((facesLabel, notfacesLabel))


    X_images = [Image.open(filename).convert('L') for filename in X]

    start = time.time()
    pred = getFaces(X_images, already_greyscale=True)
    end = time.time()
    timeTaken = end - start

    accuracy = np.equal(pred,y).astype('int').sum() / len(y)

    return (accuracy, timeTaken)

def evaluate_all_checkpoints():
    checkpoints = glob.glob('{}/*'.format(os.path.join(os.path.dirname(__file__), 'checkpoint')))
    for cp in checkpoints:
        (accuracy, timeTaken) = evaluate(cp)
        metrics.track({'checkpoint_name': cp, 'accuracy': accuracy, 'timeTaken': timeTaken})

    metrics.save()

if __name__ == '__main__':
    evaluate_all_checkpoints()