import os
from google_images_download import google_images_download
import string
import random
import glob
import cv2

images_for_nofaces_path = os.path.join(os.path.dirname(__file__), '../data/temp/images_for_nofaces')

faceCascade = cv2.CascadeClassifier('/Users/RUBI349500/Projects/ML/pytorch/face-detection/scripts/haarcascade_frontal.xml')


def remove_empty_dirs():
    for folderpath in glob.glob('{}/**'.format(images_for_nofaces_path)):
        try:
            os.rmdir(folderpath)
        except OSError:
            pass

def detect_faces(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces


def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for m in range(length))


def download(keyword=None, limit=10):
    # Download images
    if keyword is None:
        keyword = random_string(4)
    print('Getting images with keyword: {}'.format(keyword))

    response = google_images_download.googleimagesdownload()
    response.download({'keywords': keyword, 'limit': limit,
                       'output_directory': images_for_nofaces_path, 'format': 'jpg', 'type': 'photo' })
    return limit


def remove_with_faces():
    files = glob.glob('{}/**/*.jpg'.format(images_for_nofaces_path), recursive=True) + glob.glob('{}/**/*.jpeg'.format(images_for_nofaces_path), recursive=True)
    for filename in files:
        print(filename)
        try:
            faces = detect_faces(filename)
            if len(faces) > 0:
                print('FOUND FACES: Removing file {}'.format(filename))
                os.remove(filename)
        except:
            print('Removing file {}'.format(filename))
            os.remove(filename)


def download_from_keywords_file(limit=20):
    with open(os.path.join(os.path.dirname(__file__), 'keywords.txt'), 'r') as f:
        for line in f.readlines():
            download(line, limit=limit)

def collect():
    download()
    remove_with_faces()
    remove_empty_dirs()