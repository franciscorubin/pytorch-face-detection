import os
from google_images_download import google_images_download
import string
import random
import glob
import cv2
from PIL import Image
import random

images_for_nofaces_path = os.path.join(os.path.dirname(__file__), '../data/temp/images_for_nofaces')

faceCascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 'haarcascade_frontal.xml'))

crops_path = os.path.join(os.path.dirname(__file__), '../data/extracted_nofaces')

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
        except Exception as err:
            print('Removing file, Error: {}'.format(err))
            os.remove(filename)


def download_from_keywords_file(limit=100):
    with open(os.path.join(os.path.dirname(__file__), 'keywords.txt'), 'r') as f:
        for line in f.readlines():
            download(line, limit=limit)



def extract_crops(filename, w_range=(43, 7122), h_range=(55, 8984), amount=100):
    image = Image.open(filename)
    img_width, img_height = image.size
    img_name = filename.split('/')[-1]

    max_w = img_width if img_width < w_range[1] else w_range[1]
    max_h = img_height if img_height < h_range[1] else h_range[1]
    min_w = w_range[0]
    min_h = h_range[0]

    for i in range(0, amount):
        w = random.randrange(min_w, max_w)
        h = random.randrange(min_h, max_h)

        x = random.randrange(0, img_width - w)
        y = random.randrange(0, img_height - h)
  
        region_img = image.crop((x, y, x+w, y+h))

        region_img.save('{}/{}_{}.jpg'.format(crops_path, img_name, i))


def extract_crops_all_images(w_range=(43, 7122), h_range=(55, 8984), amount=100):
    files = glob.glob('{}/**/*.jpg'.format(images_for_nofaces_path), recursive=True) + glob.glob('{}/**/*.jpeg'.format(images_for_nofaces_path), recursive=True)
    for filename in files:
        print(filename)
        try:
            extract_crops(filename, w_range, h_range, amount)            
            print('Extracted crops on {}. Removing...'.format(filename))
            os.remove(filename)
        except Exception as err:
            print('Error extracting crops on {}: {}'.format(filename, err))




def collect():
    download_from_keywords_file()
    remove_with_faces()
    extract_crops_all_images()
    remove_empty_dirs()