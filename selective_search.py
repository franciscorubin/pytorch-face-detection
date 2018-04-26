import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
from PIL import Image

MIN_PIXELS = 500
MIN_WIDTH = 10
MIN_HEIGHT = 15

def get_bounding_boxes(img):
    _, regions = selectivesearch.selective_search(img, scale=300, sigma=0.8, min_size=10)

    boxes = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in boxes:
            continue
        if r['size'] < MIN_PIXELS:
            continue
        x, y, w, h = r['rect']
        if w < MIN_WIDTH:
            continue
        if h < MIN_HEIGHT:
            continue

        boxes.add(r['rect'])

    print('Number of boxes: {}'.format(len(boxes)))
    return boxes

if __name__ == "__main__":
    img = Image.open(os.path.join(os.path.dirname(__file__), 'data/test/1.jpg'))
    img = np.array(img)
    boxes = get_bounding_boxes(img)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in img:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()