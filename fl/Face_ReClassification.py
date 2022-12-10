'''

-----   변형시키는중   -----

웹캠에서 실시간으로 face_recognition으로 detecting
128개의 점

'''

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
import requests
from PIL import ImageFont, ImageDraw, Image
from dateutil.parser import parse
import json

import torch
import numpy as np
from torchvision import models,transforms
import base64

from threading import Thread


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#데이터셋을 불러올때 사용할 변형정의
transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model = torch.load('OhLeeJoMoon_model.pth', map_location=torch.device('cpu'))
model.eval()

model = model.to(device)

class_names = ['Jo', 'Lee', 'Moon', 'Oh']
class_dict = {'Jo' : '조동후', 'Lee':'이승민', 'Moon':'문지현', 'Oh':'오현승'}

id_name_dict = {'조동후': '10', '이승민':'4', '문지현':'3', '오현승':'1'}


# 이미지의 이름들을 className으로 사용, 이미지들을 따로 images에 저장
path = 'img_file'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# 이미지들을 인코딩 해서 List에 담는 것.
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

Camera_Num = 1

# # 현재 날짜와 시간 csv에 저장
# def markAttendance(name):
#     with open('Attendance.csv','r+') as f :
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')

num = 0

pre_now = datetime.now()
stString = pre_now.strftime('%H:%M:%S')
now_time = parse(stString)

# Oh_dict = {'last_time' : now_time}

######### 다중
time_dict = {'Oh_time' : now_time, 'Lee_time':now_time , 'Moon_time':now_time, 'Jo_time':now_time}



encodeListKnown = findEncodings(images)
# print(type(encodeListKnown))
print('Encoding Complete')

# 웹캠에서 사용
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    img = frame

    # 속도를 위해 resize
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # 현재 프레임에서 얼굴 좌표 찾는것
    facesCurFrame = face_recognition.face_locations(imgS)
    # print(facesCurFrame[0][0])
    # print(facesCurFrame)

    # y1, x2, y2, x1 = faceLoc



    # 인코딩
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # print(type(encodeCurFrame))
    # print(type(encodeCurFrame[0]))
    # print(encodeCurFrame)

    # 비교하기
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):

        num += 1

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # print('-'*50)
        # print(matches)
        # print('-' * 50)
        #
        # print(type(encodeListKnown))
        # print(encodeListKnown)
        #
        # print(type(encodeFace))
        # print(encodeFace)
        # print(type(encodeFace))

        # encodeListKnown에는 3개의 이미지가 있기 때문에 3개가 있는 리스트로 반환됨
        # 그 중에 faceDis가 가장 낮은 것이 가장 유사한 사진
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)

        # 가장 낮은 faceDis의 인덱스 찾기
        matchIndex = np.argmin(faceDis)


        # 프레임의 넓이가 14400보다 크다면
        if (faceLoc[1] - faceLoc[3]) * (faceLoc[2] - faceLoc[0]) >= 8000:
            a = 'found'
        # 밑에 처럼 안한 이유가 지금 test 중에서 내 얼굴과 유사한 사진이
        # 없기에 print(name)을 했을떄 name이 안나옴

            if matches[matchIndex]:
                name = classNames[matchIndex]
                # print(name)
                print(class_dict[name])
                y1, x2, y2, x1 = faceLoc
                # print(x1, y1, x2, y2)

                # 나는 왜 이거 안해도 잘되지?
                # y1, x2, y2, x1 = y1*4, x2*4 ,y2*4 ,x1*4

                # bbox그리기
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 25, 0), 4)


                # 넘파이이미지를 PIL로 만드는 법
                pil_image = Image.fromarray(img)
                cropped_img = pil_image.crop((x1-20,y1-40,x2+20,y2+20))

                # 다시 넘파이로 만들어주기
                cropped_img = np.array(cropped_img)
                cv2.imwrite(f"./save_img_file/{name}_{num}.jpg", cropped_img)

                image = Image.open(f"./save_img_file/{name}_{num}.jpg")
                image = transforms_test(image).unsqueeze(0).to(device)


                with torch.no_grad():
                    outputs = model(image)
                    _, preds = torch.max(outputs, 1)
                    # print(preds)
                    print(class_dict[class_names[preds[0]]])


                if class_dict[name] == class_dict[class_names[preds[0]]]:
                    print('same!!!')

                    # now = str(datetime.now()).split('.')[0]
                    # requests.get("http://127.0.0.1:5000/reserve",
                    #              json={'name': class_dict[name], 'time': now, 'camera_number': Camera_Num})


                    # last_time = datetime.now()
                    # last_string = now.strftime('%H:%M:%S')
                    # last_time = parse(last_string)


                    now = datetime.now()
                    now_String = now.strftime('%H:%M:%S')
                    now_time = parse(now_String)

                    con_time = '00:00:03'
                    con_time = parse(con_time).time()

                    # print('-' * 50)
                    # print(now_time)
                    # print(Oh_dict['last_time'])
                    # print('-'*50)
                    # print(parse(str(now_time-Oh_dict['last_time'])).time())

                    # if parse(str(now_time - Oh_dict['last_time'])).time() >= con_time:

                    ######### 다중
                    if parse(str(now_time - time_dict[f'{class_names[preds[0]]}_time'])).time() >= con_time:


                        # class_img_path = f"./save_img_file/{name}_{num}.jpg"
                        # img=cv2.imread(class_img_path)
                        img_str = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
                        img_dict = {'img': img_str}

                        # print(type(img_dict))
                        # img_dict = json.dumps(img_dict)
                        #
                        # print('-'*50)
                        # print(type(img_dict))
                        #
                        # print(img_dict)



                        # all_dict = json.dumps('camera_number':Camera_Num,'name': class_dict[name], 'time':str(now_time.time()), 'image':img_dict)

                        # # headers = {'Content-Type': "application/json", 'charset':'utf-8'}
                        response = requests.post("http://192.168.0.182:8000/detection/info", json={'id': id_name_dict[class_dict[name]], 'camera_number':str(Camera_Num),'name': class_dict[name], 'time':str(now_time.time()), 'image':img_dict})

                        # print('*'*50)
                        print(response.text)
                        # print('*' * 50)

                    # Oh_dict['last_time'] = now_time

                    ######### 다중
                    time_dict[f'{class_names[preds[0]]}_time'] = now_time
                    # ↖ㅇㅅㅇ↗
                    # 이걸 못해? 빨리 해
                    # 이론.. 빨리 해 흡연악귀야

                os.remove(f'./save_img_file/{name}_{num}.jpg')




                # # cv2.rectangle(img, (222, 345), (407, 159), (0, 25, 0), 4)
                #
                # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                #
                # cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # 시간 같이 보내주기 위해서
                # now = str(datetime.now()).split('.')[0]
                # requests.get("http://192.168.0.42:5000/reserve", json={'name': name, 'time':now, 'camera_number':Camera_Num})

                # time.sleep(1)



        else:
            a = 'not found'
            print("not found")

                # markAttendance(name)

    try:
        fontpath = "fonts/gulim.ttc"
        font = ImageFont.truetype(fontpath, 30)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        img = cv2.flip(img, 1)

        if a == 'found':
            draw.text((10, 20), ('F_R:' + class_dict[name]), font=font, fill=(255,0,0))
            draw.text((10, 50), ('Resnet: ' + class_dict[class_names[preds[0]]]), font=font, fill=(255,0,0))


            if class_dict[name] == class_dict[class_names[preds[0]]]:
                draw.text((10, 80), 'Same!', font=font, fill=(0, 0, 255))

            else:
                draw.text((10, 80), 'Not Same', font=font, fill=(0, 0, 255))
        else:
            draw.text((10, 60), 'Not found', font=font, fill=(0, 0, 255))

        img = np.array(img_pil)

    except:
        pass
        # fontpath = "fonts/gulim.ttc"
        # font = ImageFont.truetype(fontpath, 30)
        # img_pil = Image.fromarray(img)
        # draw = ImageDraw.Draw(img_pil)
        #
        # img = cv2.flip(img, 1)
        #
        # draw.text((10, 40), 'Not Found', font=font, fill=(0, 0, 255))
        #
        # img = np.array(img_pil)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(5) == 27:
        break
