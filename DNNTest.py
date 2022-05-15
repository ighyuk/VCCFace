# 필요한 패키지 import
from imutils.video import FPS
import imutils  # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time  # 시간 처리 모듈
import argparse  # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import numpy as np  # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import cv2  # opencv 모듈
import os

from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
import glob
face_cascade = cv2.CascadeClassifier(
    'C:/Users/Owner/PycharmProjects/FA/VGGFace/haarcascade_frontalface_alt2.xml')
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
model.load_state_dict(torch.load('gwangsu_predict.pt'))
model.eval()

path='C:/Users/Owner/PycharmProjects/FA/LastVGG/Train'
subfolders = [ os.path.basename(f) for f in os.scandir(path) if f.is_dir() ] #폴더의 이름으로 class_names 자동으로 만드는 코드
class_names=[]
class_name={}
for idx in range(len(subfolders)):
    class_name=subfolders[idx]
    class_names.append(class_name)
# print(class_names)
def face_extractor(frame):
    result = ''
    person_dict={}
    list=[]
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if faces is None:
        cv2.putText(frame, "There is no face in the frame", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face = frame[y:y + h, x:x + w]
        if type(face) is np.ndarray:
            face = cv2.resize(face, (224, 224))
            im = Image.fromarray(face, 'RGB')
            img_array = np.array(im)
            # img_array = np.expand_dims(img_array, axis=0)
            cv2.imwrite('framex.jpg', img_array)
            image = cv2.imread('framex.jpg')
            # cv2.imshow('frame', frame)
            # Preprocess image
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
            img = tfms(Image.fromarray(image)).unsqueeze(0)

            with torch.no_grad():
                # print(img)
                # imshow(img)
                # img = img.cuda()
                # outputs = model.to("cpu")
                outputs = model(img)

            # Print predictions
            # plt.imshow(test_image)
            for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist(): #주어진 output중 가장 큰 k 개의 요소반환
                prob = torch.softmax(outputs, dim=1)[0, idx].item()
                if (prob * 100 > 50):
                    result = result + '[ {} : {p:.2f}% : x = {x:d} y = {y:d} x2 = {x1:d} y2 = {y1:d} ]'.format(class_names[idx], p=prob * 100,x=x,y=y,x1=x+w,y1=y+h)
                    person_dict={'idx': idx, 'name':class_names[idx],'x':x,'y':y,'x2':x+w,'y2':y+h}
                    list.append(person_dict)
                    print(person_dict) #person_dict은 사람의 인덱스, 이름, 좌표정보 dictionary 정보(나중에 버튼만들때 쓸것)
                    cv2.putText(frame, class_names[idx], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                else:
                    result = result + '[ unknown ]'
                    cv2.putText(frame, 'unknown', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    print(result)


# # 실행을 할 때 인자값 추가
# ap = argparse.ArgumentParser()  # 인자값을 받을 인스턴스 생성
# # 입력받을 인자값 등록
# ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
# ap.add_argument("-o", "--output", type=str, help="output 비디오 경로")  # 비디오 저장 경로
# # 입력받은 인자값을 args에 저장
# args = vars(ap.parse_args())

# 얼굴 인식 모델 로드
print("[얼굴 인식 모델 로딩]")
face_detector = "./face_detector/"
prototxt = face_detector + "deploy.prototxt"  # prototxt 파일 : 모델의 레이어 구성 및 속성 정의
weights = face_detector + "res10_300x300_ssd_iter_140000.caffemodel"  # caffemodel 파일 : 얼굴 인식을 위해 ResNet 기본 네트워크를 사용하는 SSD(Single Shot Detector) 프레임워크를 통해 사전 훈련된 모델 가중치 사용
net = cv2.dnn.readNet(prototxt, weights)  # cv2.dnn.readNet() : 네트워크를 메모리에 로드

# # input 비디오 경로가 제공되지 않은 경우 webcam
# if not args.get("input", False):
#     print("[webcam 시작]")
#     vs = cv2.VideoCapture(0)
#
# # input 비디오 경로가 제공된 경우 video
# else:
#     print("[video 시작]")
#     vs = cv2.VideoCapture(args["input"])
# url = "http://hye:225@192.168.219.118:8080/video"
url = "http://hye:225@117.20.203.79:8080/video"#포트포워드혜원
# url = "http://dd:1234@59.24.115.118:8080/video"#포트포워도은
# Doing some Face Recognition with the webcam
vs = cv2.VideoCapture(url)
vs.set(cv2.CAP_PROP_FPS, 5)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 240) # 가로
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 280) # 세로
# 인식할 최소 확률
minimum_confidence = 0.7

writer = None

# fps 정보 초기화
fps = FPS().start()

# 비디오 스트림 프레임 반복
while True:
    result = ''
    person_dict = {}
    list = []
    # 프레임 읽기
    ret, frame = vs.read()

    # # 읽은 프레임이 없는 경우 종료
    # if args["input"] is not None and frame is None:
    #     break

    # 프레임 resize
    frame = imutils.resize(frame, width=400)

    # 이미지 크기
    (H, W) = frame.shape[:2]

    # blob 이미지 생성
    # 파라미터
    # 1) image : 사용할 이미지
    # 2) scalefactor : 이미지 크기 비율 지정
    # 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 얼굴 인식
    net.setInput(blob)  # setInput() : blob 이미지를 네트워크의 입력으로 설정
    detections = net.forward()  # forward() : 네트워크 실행(얼굴 인식)

    # 얼굴 번호
    number = 0

    # 얼굴 인식을 위한 반복
    for i in range(0, detections.shape[2]):
        # 얼굴 인식 확률 추출
        confidence = detections[0, 0, i, 2]


        # 얼굴 인식 확률이 최소 확률보다 큰 경우
        if confidence > minimum_confidence:
            # bounding box 위치 계산
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            # bounding box 가 전체 좌표 내에 있는지 확인
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(W - 1, endX), min(H - 1, endY))

            # cv2.putText(frame, "Face[{}]".format(number + 1), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0), 2)  # 얼굴 번호 출력
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # bounding box 출력
            face = frame[startY:endY, startX:endX]
            if type(face) is np.ndarray and len(face) != 0:
                # print(face.shape)
                # print(face.ndim)
                # print(len(face))
                face = cv2.resize(face, (224, 224))
                im = Image.fromarray(face, 'RGB')
                img_array = np.array(im)
                # img_array = np.expand_dims(img_array, axis=0)
                cv2.imwrite('framex.jpg', img_array)
                image = cv2.imread('framex.jpg')
                # cv2.imshow('frame', frame)
                # Preprocess image
                tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
                img = tfms(Image.fromarray(image)).unsqueeze(0)

                with torch.no_grad():
                    # print(img)
                    # imshow(img)
                    # img = img.cuda()
                    # outputs = model.to("cpu")
                    outputs = model(img)

                # Print predictions
                # plt.imshow(test_image)
                for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():  # 주어진 output중 가장 큰 k 개의 요소반환
                    prob = torch.softmax(outputs, dim=1)[0, idx].item()
                    if (prob * 100 > 50):
                        result = result + '[ {} : {p:.2f}% : x = {x:d} y = {y:d} x2 = {x1:d} y2 = {y1:d} ]'.format(
                            class_names[idx], p=prob * 100, x=startX, y=startY, x1=endX, y1=endY)
                        person_dict = {'idx': idx, 'name': class_names[idx], 'x': startX, 'y': startY, 'x2': endX, 'y2': endY}
                        list.append(person_dict)
                        print(person_dict)  # person_dict은 사람의 인덱스, 이름, 좌표정보 dictionary 정보(나중에 버튼만들때 쓸것)
                        cv2.putText(frame, class_names[idx], (startX, startY), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                    else:
                        result = result + '[ unknown ]'
                        cv2.putText(frame, 'unknown', (startX, startY), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "There is no face in the frame", (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 255, 0), 2)
            # number = number + 1  # 얼굴 번호 증가
        # else:
        #     cv2.putText(frame, "There is no face in the frame", (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        print(result)

    # 프레임 resize
    frame = imutils.resize(frame, width=500)

    # 프레임 출력
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키를 입력하면 종료
    if key == ord("q"):
        break

    # fps 정보 업데이트
    fps.update()

    # output video 설정
    # if args["output"] != "" and writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    #
    # # 비디오 저장
    # if writer is not None:
    #     writer.write(frame)

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 종료
vs.release()
cv2.destroyAllWindows()