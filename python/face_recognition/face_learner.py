# ======================================================================
# Project Name    : Face Identify 
# File Name       : face_lerner.py
# Encoding        : utf-8
# Creation Date   : 2021/02/20
# ======================================================================

import os
import re
import numpy as np
import time
import glob
import shutil
import PIL.Image
from PIL import ImageEnhance
import subprocess
import cv2

def takePic(cascade, picnum_max):
    """take pictures for learning

    """
    cap = cv2.VideoCapture(0)
    color = (255,255,255)
    picture_num = 1
    while True:
        ret, frame = cap.read()
        facerect = cascade.detectMultiScale(frame, scaleFactor=1.7, minNeighbors=4, minSize=(100,100))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(picture_num), (10,500), font, 4,(0,0,0),2,cv2.LINE_AA)
        if len(facerect) > 0:
            for (x,y,w,h) in facerect:
                picture_name = './pos/pic' + str(picture_num) + '.jpg'
                cv2.imwrite(picture_name, frame)
                picture_num += 1
        cv2.imshow("frame", frame)
        if picture_num == picnum_max + 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def removePic():
    """remove files for initialize

    """
    os.chdir('cascade')
    for x in glob.glob('*.xml'):
        os.remove(x)
    os.chdir('../')

    os.chdir('pos')
    for x in glob.glob('*.jpg'):
        os.remove(x)
    os.chdir('../')

    if os.path.exists('pos.txt'):
        os.remove('pos.txt')

def countPic():
    """count a number of taken pictures

    """

    files = os.listdir("./pos")
    count = 0
    for file in files:
        count = count + 1

    return count

def bulkOut():
    """Bult out pics

    """
    # a number of taken pics
    originalnum = countPic()
    # a number of present total pics
    imageNum = countPic()+1

    # Flip horizontal
    for num in range(1, originalnum + 1 ):
        fileName = './pos/pic' + str(num) + '.jpg'
        if not os.path.exists(fileName):
            continue
        img = cv2.imread(fileName)
        yAxis = cv2.flip(img, 1)
        newFileName = './pos/pic' + str(imageNum) + '.jpg'
        cv2.imwrite(newFileName,yAxis)
        imageNum += 1
    print('*** Flip horizontal is finished *** \n')

    # Change Saturation
    SATURATION = 0.5
    CONTRAST = 0.5
    BRIGHTNESS = 0.5
    SHARPNESS = 2.0

    for num in range(1, 2 * originalnum + 1):
        fileName = './pos/pic' + str(num) + '.jpg'
        if not os.path.exists(fileName): 
            continue
        img = PIL.Image.open(fileName)
        saturation_converter = ImageEnhance.Color(img)
        saturation_img = saturation_converter.enhance(SATURATION)
        newFileName = './pos/pic' + str(imageNum) + '.jpg'
        saturation_img.save(newFileName)
        imageNum += 1
    print('*** Change Saturation is finished *** \n')

    # Change Contsract
    for num in range(1, 3 * originalnum + 1):
        fileName = './pos/pic' + str(num) + '.jpg'
        if not os.path.exists(fileName):
            continue
        img = PIL.Image.open(fileName)
        contrast_converter = ImageEnhance.Contrast(img)
        contrast_img = contrast_converter.enhance(CONTRAST)
        newFileName = './pos/pic' + str(imageNum) + '.jpg'
        contrast_img.save(newFileName)
        imageNum += 1
    print('*** Change Constract is finished *** \n')

    # Change Brightness
    for num in range(1, 4 * originalnum + 1):
        fileName = './pos/pic' + str(num) + '.jpg'
        if not os.path.exists(fileName):
            continue
        img = PIL.Image.open(fileName)
        brightness_converter = ImageEnhance.Brightness(img)
        brightness_img = brightness_converter.enhance(BRIGHTNESS)
        newFileName = './pos/pic' + str(imageNum) + '.jpg'
        brightness_img.save(newFileName)
        imageNum += 1
    print('*** Change Brightness is finished *** \n')

    # Change Sharpness
    for num in range(1, 5 * originalnum + 1):
        fileName = './pos/pic' + str(num) + '.jpg'
        if not os.path.exists(fileName):
            continue
        img = PIL.Image.open(fileName)
        sharpness_converter = ImageEnhance.Sharpness(img)
        sharpness_img = sharpness_converter.enhance(SHARPNESS)
        newFileName = './pos/pic' + str(imageNum) + '.jpg'
        sharpness_img.save(newFileName)
        imageNum += 1
    print('*** Change Sharpness is finished *** \n')

    # Rotate by 15 deg.
    for num in range(1, 6 * originalnum + 1):
        fileName = './pos/pic' + str(num) + '.jpg'
        if not os.path.exists(fileName):
            continue
        # read original file
        img = cv2.imread(fileName)
        h, w = img.shape[:2]
        size = (w, h)

        # define angle to rotare
        angle = 15
        angle_rad = angle/180.0*np.pi

        # caluclate a size of pic after rotation
        w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
        h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
        size_rot = (w_rot, h_rot)

        # rotate 
        center = (w/2, h/2)
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # add translation)
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
        affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2

        img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)

        cv2.imwrite(newFileName, img_rot)
        newFileName = './pos/pic' + str(imageNum) + '.jpg'
        saturation_img.save(newFileName)
        imageNum += 1
    print('*** Rotation by 15 deg. is finished *** \n')

    # Rotate by -15 deg.
    for num in range(1, 7* originalnum + 1):
        fileName = './pos/pic' + str(num) + '.jpg'
        if not os.path.exists(fileName):
            continue
        # read original file
        img = cv2.imread(fileName)
        h, w = img.shape[:2]
        size = (w, h)

        # define angle to rotare
        angle = -15
        angle_rad = angle/180.0*np.pi

        # caluclate a size of pic after rotation
        w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
        h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
        size_rot = (w_rot, h_rot)

        # rotate 
        center = (w/2, h/2)
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # add translation)
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
        affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2

        img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)

        cv2.imwrite(newFileName, img_rot)
        newFileName = './pos/pic' + str(imageNum) + '.jpg'
        saturation_img.save(newFileName)
        imageNum += 1
    print('*** Rotation by -15 deg. is finished ***\n')

    print('*** Bulking out is completed ***\n')

def generatePosFile(cascade):
    """make text file of face positions in pictures 

    """
    fpos = open('pos.txt', 'a')
    for fileName in glob.glob('./pos/*.jpg'):
        img = cv2.imread(fileName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            text =  fileName + ' 1 ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
            fpos.write(text)
    print('*** making pos.txt is finished ***')

# user_fileName = input('Please input your fileName\n')
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
picnum_max = input('please input a number of pictures to take\n')

# remove pos and files in pic/pos
removePic()

# start video and take pictures
takePic(cascade, int(picnum_max))

# count a number of picutres
bulkOut()

# make text file of face positions in pictures 
generatePosFile(cascade)

subprocess.call('opencv_createsamples -info pos.txt -vec pos.vec -num ' + str(countPic()), shell=True) 

posnum = input('please input a number of created pos\n')
subprocess.call('opencv_traincascade -data ./cascade -vec pos.vec -bg neg.txt -numPos ' + posnum + ' -numNeg 40', shell=True)

