import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# pkl파일 불러오기 위해서 사용하기 위해
import joblib
import pandas as pd
model = joblib.load('body.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        try:
            body = results.pose_landmarks.landmark
            # print(body)
            body_list = []
            for temp in body:
                body_list.append([temp.x, temp.y, temp.z, temp.visibility])

            body_row = list(np.array(body_list).flatten())

            #### 모델 예측 ####0
            X = pd.DataFrame([body_row])
            print(model.predict(X)[0])

            if os.path.isfile('coords.csv') == False:  # 만약 파일이 없으면
                print('-'*50)
                print(os.path.isfile('coords.csv'))
                print('-' * 50)
                landmarks = ['label']

                for temp in range(1, len(body) + 1):

                    # csv에 column명
                    # x1, y1, z1 이런식으로 들어감
                    landmarks += ['x{}'.format(temp), 'y{}'.format(temp), 'z{}'.format(temp), 'v{}'.format(temp)]

                # print(landmarks)

                with open('coords.csv', mode='w', newline='') as f:

                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks)
                    f.close()

            else:
                if cv2.waitKey(10) & 0xFF == ord('s'):
                    body_row.insert(0, 'stand')  # 0번째 column에 stand 집어넣기
                    with open('coords.csv', mode='a', newline='') as f:  # 계속 추가할거기 때문에 mode='a'
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(body_row)
                        f.close()

                elif cv2.waitKey(10) & 0xFF == ord('d'):
                    body_row.insert(0, 'sit')  # 0번째 column에 sit 집어넣기
                    with open('coords.csv', mode='a', newline='') as f:  # 계속 추가할거기 때문에 mode='a'
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(body_row)
                        f.close()

                elif cv2.waitKey(10) & 0xFF == ord('f'):
                    body_row.insert(0, 'falldown')  # 0번째 column에 falldown 집어넣기
                    with open('coords.csv', mode='a', newline='') as f:  # 계속 추가할거기 때문에 mode='a'
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(body_row)
                        f.close()





        except:
            pass

        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()