import os
import glob
import cv2
from PIL import Image

celebA_source_path = os.path.join(os.path.dirname(__file__), '../data/celebA/img_celeba')
celebA_extracted_faces_path = os.path.join(os.path.dirname(__file__), '../data/extracted_faces')

faceCascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 'haarcascade_frontal.xml'))

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces



def traverse_all_images():
    files = glob.glob('{}/*'.format(celebA_source_path))
    done = 0
    failed = 0
    for filename in files:
        img = cv2.imread(filename)
        img_pil = Image.open(filename)
        img_name = filename.split('/')[-1]
        try:
            faces = detect_faces(img)
            if len(faces) > 1 or len(faces) <= 0:
                print('Invalid number of faces found ({}). Skipping.'.format(len(faces)))
                failed+=1
            else:
                (x, y, w, h) = faces[0]
                face_region = img_pil.crop((x, y, x+w, y+h))
                face_region.save('{}/{}.jpg'.format(celebA_extracted_faces_path, img_name))
                done+=1
        except Exception as err:
            print('Error extracting crops on {}: {}'.format(filename, err))
            failed+=1

        print('Total: {}, Saved: {}, Failed: {}'.format(done+failed, done, failed))


if __name__ == '__main__':
    traverse_all_images()