import os

from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
import glob
face_cascade = cv2.CascadeClassifier(
    'C:/projects/VCCFace/haarcascade_frontalface_alt2.xml')
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
model.load_state_dict(torch.load('gwangsu_predict.pt'))
model.eval()

path='C:/projects/VCCFace/Train'
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
            # print(face.shape)
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


    # return cropped_face
# class_names = {
#     "0": "Bill",
#     "1": "Elon",
#     "2": "Jihyo",
#     "3": "Sana",
#     "4": "Suji"
# }










video_capture = cv2.VideoCapture('C:/projects/VCCFace/jihyo.mp4')
video_capture.set(cv2.CAP_PROP_FPS, 5)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 240) # 가로
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 280) # 세로
while True:
    _, frame = video_capture.read()
    face_extractor(frame)



        # print('-----')

    # else:
    #     cv2.putText(frame, "There is no face in the frame", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    resize_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Video', resize_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
