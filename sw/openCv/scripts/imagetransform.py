import subprocess
import sys
import os
from PIL import Image
import PIL.ImageOps
import cv2
import pywt
import pywt.data
import numpy as np
import argparse

def is_jpg(name):
    return name.endswith('.jpg')

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
def apply_many(path):
    output = os.path.join(os.path.split(path)[0], 'all')
    make_dir(output)
    f = os.path.split(path)[1]
    print('applying all transforms to {}'.format(f))
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    erosion = cv2.erode(image, np.ones((21, 21)))

    blurred = cv2.blur(erosion, (21, 21))

    inverted = np.invert(blurred)

    cv2.imwrite(output + os.sep + f.replace('.jpg', '') + '_all.jpg', inverted)   

def erode_boundaries(path):
    output = os.path.join(os.path.split(path)[0], 'eroded')
    make_dir(output)
    f = os.path.split(path)[1]
    print('eroding boundaries of {}'.format(f))
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    erosion = cv2.erode(image, np.ones((21, 21)))
    cv2.imwrite(output + os.sep + f.replace('.jpg', '') + '_eroded.jpg', erosion)

def wavelet_transform(path):
    output = os.path.join(os.path.split(path)[0], 'wavelet')
    make_dir(output)
    f = os.path.split(path)[1]
    print('applying wavelet transform to {}'.format(f))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    b,g,r = cv2.split(img)
    # no idea what this does, got it off stackoverflow
    coeffsb = pywt.dwt2(b, 'haar')
    LLb, (LHb, HLb, HHb) = coeffsb

    coeffsg = pywt.dwt2(g, 'haar')
    LLg, (LHg, HLg, HHg) = coeffsg

    coeffsr = pywt.dwt2(r, 'haar')
    LLr, (LHr, HLr, HHb) = coeffsr

    combined = cv2.merge((LLb, LLg, LLr))
    grey = cv2.cvtColor(combined.astype('float32'), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output + os.sep + f.replace('.jpg', '') + '_wavelet.jpg', grey)

def blur_images(path):
    output = os.path.join(os.path.split(path)[0], 'blurred')
    make_dir(output)
    f = os.path.split(path)[1]
    print('blurring {}'.format(f))
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    blurred = cv2.blur(image, (21, 21))
    cv2.imwrite(output + os.sep + f.replace('.jpg', '') + '_blurred.jpg', blurred)

def flip_colors(path):
    output = os.path.join(os.path.split(path)[0], 'inverse')
    make_dir(output)
    f = os.path.split(path)[1]
    print('inverting colors of {}'.format(f))
    image = Image.open(path)
    inversion = PIL.ImageOps.invert(image)
    inversion.save(output + os.sep + f.replace('.jpg', '') + '_inverted.jpg')

def reduce_resolution(path):
    output = os.path.join(os.path.split(path)[0], 'reduced')
    make_dir(output)
    f = os.path.split(path)[1]
    print('reducing resolution of {}'.format(f))
    image = Image.open(path)
    reduction = image.resize((int(image.size[0] / 4), int(image.size[1] / 4)))
    output_file_path = output + os.sep + f
    reduction.save(output_file_path)
    return output_file_path

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    args = parser.parse_args()

    base_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], shell = True).decode("utf-8").strip()
    if 'win32' in sys.platform:
        base_dir = base_dir.replace('/', '\\')
        
    if args.path == None:
        images_folder = os.path.join(base_dir, 'sw', 'openCv', 'images')
    else:
        images_folder = args.path

    import time
    s_time = time.time()

    for f in os.listdir(images_folder):
        image_path = images_folder + os.sep + f
        if os.path.isfile(image_path) and is_jpg(f):
            reduced_img = reduce_resolution(image_path)
            flip_colors(reduced_img)
            blur_images(reduced_img)
            wavelet_transform(reduced_img)
            erode_boundaries(reduced_img)
            apply_many(reduced_img)

    e_time = time.time()
    print("finished in {} seconds".format(e_time - s_time))


if __name__ in '__main__':
    main()