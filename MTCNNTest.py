import os

from mtcnn.mtcnn import MTCNN
import cv2 as cv
from matplotlib import pyplot
from matplotlib.patches import Rectangle
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

path = 'C:/Users/Owner/PycharmProjects/FA/LastVGG/Train'
subfolders = [os.path.basename(f) for f in os.scandir(path) if f.is_dir()]  # 폴더의 이름으로 class_names 자동으로 만드는 코드
class_names = []
class_name = {}
for idx in range(len(subfolders)):
    class_name = subfolders[idx]
    class_names.append(class_name)


# print(class_names)
# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    # show the plot
    pyplot.show()


def face_extractor(frame):
    result = ''
    person_dict = {}
    list = []
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    cv2.imwrite('framex.jpg', frame)
    filename = 'framex.jpg'
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    faces = detector.detect_faces(pixels)

    # # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if faces is None:
        cv2.putText(frame, "There is no face in the frame", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Crop all faces found
    for face in faces:
        # print(face)
        x, y, width, height = face['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw_image_with_boxes(filename, faces)
        # face = frame[y:y + h, x:x + w]
        if type(rect) is np.ndarray:
            face = cv2.resize(rect, (224, 224))
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
                        class_names[idx], p=prob * 100, x=x, y=y, x1=x + w, y1=y + h)
                    person_dict = {'idx': idx, 'name': class_names[idx], 'x': x, 'y': y, 'x2': x + w, 'y2': y + h}
                    list.append(person_dict)
                    print(person_dict)  # person_dict은 사람의 인덱스, 이름, 좌표정보 dictionary 정보(나중에 버튼만들때 쓸것)
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


# url = "http://hye:225@192.168.219.118:8080/video"
url = "http://hye:225@117.20.203.79:8080/video"  # 포트포워드혜원
# url = "http://dd:1234@59.24.115.118:8080/video"#포트포워도은
# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(url)
video_capture.set(cv2.CAP_PROP_FPS, 5)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 240)  # 가로
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)  # 세로
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
video_capture.release()
cv2.destroyAllWindows()
