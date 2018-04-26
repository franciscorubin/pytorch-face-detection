import os
from PIL import Image, ImageDraw
import sys
import random
from model import Model
from face_classifier import isFace
import numpy as np

region_height = 24
region_width = 24


class FaceDetector(object):
    def __init__(self, imgPath):
        self.base_img = Image.open(imgPath)
        self.drawn_img = self.base_img.copy()
        self.draw = ImageDraw.Draw(self.drawn_img)

        self.base_img = self.base_img.convert('L')

    def paint_faces(self):
        detections = self.detect_faces()
        for det in detections:
            self.paint_rectangle(det)

    def paint_rectangle(self, rectangle):
        (zoom, x, y) = rectangle
        self.draw.rectangle((x*zoom, y*zoom, (x+24)*zoom, (y+24)*zoom), outline='red')

    def base_img_zoomed(self, zoom):
        w, h = self.base_img.size
        return self.base_img.resize((int(w / zoom), int(h / zoom)))

    def should_include_region(self, region_img):
        if isFace(self.normalize(region_img)):
          return True
        return False

    def normalize(self, region_img):
        #TODO make sure normalization matches training normalization        
        img = np.array(region_img)
        mean = img.mean()
        std = img.std()
        return (img - mean) / std

    def detect_faces(self):
        img_width, img_height = self.base_img.size
        regions = []
        detections = []

        total_checks = 0
        zoom = 1
        zoomed_img = self.base_img
        # Stop zooming when one of the image dimensions is lower than the region size
        while img_width >= region_width and img_height >= region_height:
            x_index = 0
            y_index = 0

            while y_index + region_height <= img_height:
                while x_index + region_width <= img_width:
                    region_img = zoomed_img.crop((x_index, y_index, x_index+region_width, y_index+region_height))
                    if self.should_include_region(region_img):
                        rectangle = (zoom, x_index, y_index)
                        detections.append(rectangle)

                    total_checks += 1
                    x_index += 2  # TODO analyse best increases here

                x_index = 0
                y_index += 2  # TODO analyse best increases here

            x_index = 0
            y_index = 0
            zoom += 1  # TODO analyse best increases here
            zoomed_img = self.base_img_zoomed(zoom)
            img_width, img_height = zoomed_img.size

        print('Total checks: {}'.format(total_checks))
        print('Total detections: {}'.format(len(detections)))
        return detections


if __name__ == '__main__':
    testImage = os.path.join(os.path.dirname(__file__), './data/test/1.jpg')
    detector = FaceDetector(testImage)
    detector.paint_faces()
    detector.drawn_img.show()