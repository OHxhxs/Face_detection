from flask import Flask,request,jsonify, Response
import cv2
import face_recognition
import os
import numpy as np
import json

import requests
import base64

# 스케줄 설정
import time
from apscheduler.schedulers.background import BackgroundScheduler

# 이미지 주소에서 다운로드
import urllib.request

# path = 'C:/Users/HP/Desktop/PycharmProject/Sochket_demo/fl/img_file'
# images = []
# classNames = []
# myList = os.listdir(path)
# # print(myList)
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)
#
# # 이미지들을 인코딩 해서 List에 담는 것.
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#
#     return encodeList

# encodeListKnown = findEncodings(images)
# print('Encoding Complete')

def create_app():
    app = Flask(__name__)

    # 기본url에 나오는 것
    @app.route('/')


    def index():
        return 'Hello_World'

    @app.route('/reserve', methods=['POST'])
    def reserve():
        data = request.get_json()

        img_example = json.loads(data['image'])
        data = base64.b64decode(img_example['img'])
        jpg_arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
        print(img.shape)
        cv2.imwrite('get.jpg', img)

    @app.route('/images', methods=['POST'])
    def Imagess():
        data = request.get_json()
        print(data)

        return "Success"

        # img = base64.b64decode(data)
        # jpg_arr = np.frombuffer(data, dtype=np.uint8)
        # img = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
        #
        # cv2.imwrite('get.jpg', img)
        # print(data)



        return "done"

    @app.route('/save_img')
    def Save_Img():
        # data = request.get_json()

        # {"patientCode" : "ID값", "patientName" : "이름", "patientPicture" : "이미지 경로" }
        r = requests.post("http://192.168.0.182:8000/patient/info")
        data = r.json()
        print(data)
        patientCode = data[0]['patientCode']
        patientName = data[0]['patientName']
        patientImg_url = data[0]['patientPicture']

        # url = "다운받으려는 이미지의 url주소를 입력하세요"
        savelocation = f"C:/Users/HP/Desktop/PycharmProject/Sochket_demo/fl/get_image/{patientName}.jpg"  # 내컴퓨터의 저장 위치
        urllib.request.urlretrieve('http://192.168.0.182:8000' + patientImg_url, savelocation)
        return "Done"

    # # 스케쥴 설정
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(func=Save_Img, trigger="interval", seconds=3)
    # scheduler.start()

    #1분마다
    # sched_result = sched.add_job(Save_Img, 'cron', minute='*/1')

    return app