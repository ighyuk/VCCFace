import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
import PIL.ImageOps
from torchvision import transforms

num_augmented_images = 10
file_path = 'C:/projects/VCCFace/Train/'
file_names = os.listdir(file_path) # 특정폴더만 하고싶다면 파일_네임즈을 ['doeun', 'hyewon']처럼 바꿈
augment_cnt = 1

for ii in file_names:
    file_path_2 = file_path + ii + '/'
    file_names_2 = os.listdir(file_path_2)
    total_origin_image_num = len(file_names_2)

    for i in range(0, num_augmented_images): # a ~ b-1 사용처는 특별히 없음 넘 어그먼티드이미지만큼 반복
        change_picture_index = random.randrange(0, total_origin_image_num) # a ~ b-1 사용처는 사진고르기
        print(change_picture_index)
        print(file_names_2[change_picture_index])
        file_name = file_names_2[change_picture_index]

        origin_image_path = file_path_2 + file_name
        print(origin_image_path)
        image = Image.open(origin_image_path)
        random_augment = random.randrange(1, 4)

        if (random_augment == 1):
            # 이미지 좌우 반전
            print("invert")
            inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            inverted_image.save(file_path_2 + 'inverted_' + str(augment_cnt) + '.png')

        elif (random_augment == 2):
            # 이미지 기울이기
            print("rotate")
            rotated_image = image.rotate(random.randrange(-20, 20))
            rotated_image.save(file_path_2 + 'rotated_' + str(augment_cnt) + '.png')

        elif (random_augment == 3):
            # 노이즈 추가하기
            img = cv2.imread(origin_image_path)
            print("noise")
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy_array = img + gauss
            noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
            noisy_image.save(file_path_2 + 'noiseAdded_' + str(augment_cnt) + '.png')

        augment_cnt += 1