import subprocess
import re
import os
import glob

imgs_path = os.path.join(os.path.dirname(__file__), '../data/celebA/img_celeba')

def get_image_size(filename):
    return list(map(int, re.findall('(\d+)x(\d+)', subprocess.getoutput("file " + filename))[-1]))


max_w = 0
min_w = 999999999
max_h = 0
min_h = 999999999
files = glob.glob('{}/*.jpg'.format(imgs_path))
for fl in files:
    (w, h) = get_image_size(fl)
    if w > max_w:
        max_w = w
    if w < min_w:
        min_w = w
    if h > max_h:
        max_h = h
    if h < min_h:
        min_h = h

print('max_w: {}'.format(max_w))
print('min_w: {}'.format(min_w))
print('max_h: {}'.format(max_h))
print('min_h: {}'.format(min_h))


# Result:

# max_w: 7122
# min_w: 43
# max_h: 8984
# min_h: 55