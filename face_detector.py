import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from PIL import Image, ImageDraw
import sys
import random
from model import Model
from face_classifier import isFace, getFaces, init
import numpy as np
from pytorch_utils.helpers.progress_bar import print_to_tqdm
import time
import argparse


parser = argparse.ArgumentParser(description='Face Detection')
parser.add_argument('--image', '-i', default=None, type=str, help='image chosen to detect faces')
parser.add_argument('--frompath', '-f', default=None, type=str, help='path from which to get images')
parser.add_argument('--topath', '-t', default=None, type=str, help='path to save images')

args = parser.parse_args()

region_height = 24
region_width = 24

class FaceDetector(object):
    def __init__(self, imgPath, max_image_size=500, index_increase=2, zoom_increase=1):
        self.base_img = Image.open(imgPath)
        self.drawn_img = self.base_img.copy()
        self.draw = ImageDraw.Draw(self.drawn_img)

        self.index_increase = index_increase
        self.zoom_increase = zoom_increase

        self.base_img = self.base_img.convert('L')
        self.scale = 1
        self.resize_img(max_image_size)

    def process(self):
        start = time.time()
        (crops_images, crops_info) = self.generate_crops()
        endCrops = time.time()
        detections = self.detect_faces(crops_images, crops_info)
        endDetector = time.time()
        self.paint_faces(detections)
        print('Time taken (seconds): Crops: {}, Detection: {}, Total: {}'.format(endCrops - start, endDetector - endCrops, endDetector - start))

    def resize_img(self, max_size=500):
        (width, height) = self.base_img.size
        self.scale = max_size / max(width, height) if max(width, height) > max_size else 1
        self.base_img = self.base_img.resize(map(int, self.scale*np.array(self.base_img.size)), Image.ANTIALIAS)

    def paint_faces(self, detections):
        for det in detections:
            self.paint_rectangle(det)

    def paint_rectangle(self, rectangle):
        (zoom, x, y) = rectangle
        self.draw.rectangle(((1/self.scale)*x*zoom, (1/self.scale)*y*zoom, (1/self.scale)*(x+24)*zoom, (1/self.scale)*(y+24)*zoom), outline='red')

    def base_img_zoomed(self, zoom):
        w, h = self.base_img.size
        return self.base_img.resize((int(w / zoom), int(h / zoom)))

    def should_include_region(self, region_img):
        if isFace(region_img, already_greyscale=True):
          return True
        return False

    def detect_faces(self, crops_images, crops_info):
        faces = getFaces(crops_images, already_greyscale=True)
        return np.array(crops_info)[faces]

    def generate_crops(self):
        img_width, img_height = self.base_img.size

        zoom = 1
        zoomed_img = self.base_img

        crops_info = []
        crops_images = []
        # Stop zooming when one of the image dimensions is lower than the region size
        while img_width >= region_width and img_height >= region_height:
            x_index = 0
            y_index = 0

            while y_index + region_height <= img_height:
                while x_index + region_width <= img_width:
                    region_img = zoomed_img.crop((x_index, y_index, x_index+region_width, y_index+region_height))
                    rectangle = (zoom, x_index, y_index)
                    crops_info.append(rectangle)
                    crops_images.append(region_img)

                    x_index += self.index_increase  # TODO analyse best increases here

                x_index = 0
                y_index += self.index_increase  # TODO analyse best increases here

            x_index = 0
            y_index = 0
            zoom += self.zoom_increase  # TODO analyse best increases here
            zoomed_img = self.base_img_zoomed(zoom)
            img_width, img_height = zoomed_img.size

        print('Total crops: {}'.format(len(crops_images)))
        return (crops_images, crops_info)

import glob
from tqdm import tqdm
def detect_from_path(frompath, topath):
    init()
    files = glob.glob('{}/**/*.jpg'.format(frompath), recursive=True)
    with print_to_tqdm() as orig_stdout:
        for file in tqdm(files, file=orig_stdout, dynamic_ncols=True):
            detector = FaceDetector(file, max_image_size=500, index_increase=3, zoom_increase=1)
            detector.process()

            filename = file.split('/')[-1]
            detector.drawn_img.save('{}/{}.jpg'.format(topath, filename))

def detect_single(file):
    init()
    detector = FaceDetector(file, max_image_size=500, index_increase=3, zoom_increase=1)
    detector.process()
    detector.drawn_img.show()

if __name__ == '__main__':
    if args.image:
        detect_single(args.image)
    elif args.frompath and args.topath:
        detect_from_path(args.frompath, args.topath)
    else:
        print('Wrong parameters supplied.')