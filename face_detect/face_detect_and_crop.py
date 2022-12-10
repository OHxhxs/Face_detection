import dlib
import cv2
import numpy as np
import json
from PIL import Image
import os

# create list for landmarks
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))


# create face detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 이미지

dir_path = './face_img_file'



for dir in os.listdir(dir_path):
    print(dir)

    for file_name in os.listdir(os.path.join(dir_path,dir)):
        print(file_name)

        image = cv2.imread(dir_path + f'/{dir}' + f'/{file_name}')
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces (up-sampling=1)
        face_detector = detector(img_gray, 1)

        print(image.shape)
        for face in face_detector:

            x1 = face.left()-int(image.shape[0]//20)
            y1 = face.top()-int(image.shape[0]//10)
            x2 = face.right()+int(image.shape[0]//20)
            y2 = face.bottom()+int(image.shape[0]//20)

            # 넘파이 이미지를 PIL로 만들고
            pil_image = Image.fromarray(image)

            # 이미지에서 얼굴 부분만 crop
            cropped_img = pil_image.crop((x1,y1,x2,y2))

            # 저장을 위해 crop한 이미지를 다시 array로
            cropped_img = np.array(cropped_img)

            cv2.imwrite(f'./save_crop_img/{dir}/{file_name}',cropped_img)
