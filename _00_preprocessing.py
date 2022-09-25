import splitfolders
import albumentations as A
import cv2
from matplotlib import pyplot as plt
import random
import os, glob, shutil
from tqdm import tqdm
import numpy as np

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def get_transforms():
    transforms = {}
    # p == Transform 적용될 확률 --> 증식을 위해 적용하는 것이므로 1로 설정
    transforms['rc'] = A.Compose([A.Resize(224,224), A.RandomCrop(width=200, height=200, p=1)])
    transforms['hf'] = A.Compose([A.HorizontalFlip(p=1)])
    transforms['bc'] = A.Compose([A.RandomBrightnessContrast(p=1),
                                  A.RandomGamma(p=1),
                                  A.CLAHE(p=1)], p=1)
    transforms['trp'] = A.Compose([A.Transpose()])
    transforms['sr'] = A.Compose([A.ShiftScaleRotate(p=1)])
    transforms['gb'] = A.Compose([A.GaussianBlur(p=1)])
    return transforms


def get_num_limit(data_path):
    result_list = []
    for dirname in os.listdir(data_path):
        file_cnt = len(os.listdir(os.path.join(data_path, dirname)))
        result_list.append((dirname, file_cnt))
    result_list.sort(key=lambda x:x[1], reverse=True)
    return abs(result_list[-1][1]-result_list[0][1])




def augment(data_path, num_limit=200):
    # Transform pipline 선언
    transforms = get_transforms()

    for class_name in os.listdir(data_path):
        file_cnt = len(os.listdir(os.path.join(data_path, class_name)))
        num_max = num_limit - file_cnt

        target_list = glob.glob(os.path.join(data_path, class_name, '**', '*.*'), recursive=True)
        random.seed(48)  # 랜덤시드 고정 (일정한 형태로)
        random.shuffle(target_list)
        saved_img_cnt = 0
        ratio = (lambda x, y: x // y if x // y >= 1 else 1)(num_max, file_cnt)
        # print(class_name, num_max, file_cnt, ratio)

        save_path = os.path.join(data_path.replace('rename', 'aug'), class_name)
        os.makedirs(save_path, exist_ok=True)

        for file in tqdm(target_list):
            # 원본 복사
            shutil.copy(file, f'{save_path}/{os.path.basename(file)}')

            if num_max > saved_img_cnt:
                try:
                    img_array = np.fromfile(file, np.uint8)
                    image = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)

                    target_keys = random.sample(transforms.keys(), ratio)
                    for key in target_keys:
                        # 증식기법 적용
                        transformed = transforms[key](image=image)
                        transformed_image = transformed["image"]

                        # visualize(transformed_image)
                        spl_filename = os.path.splitext(os.path.basename(file))
                        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

                        # 한글 경로 문제 해결
                        extension = os.path.splitext(file)[1]  # 이미지 확장자
                        result, encoded_img = cv2.imencode(extension, transformed_image)    # 저장할 확장자로 인코딩

                        if result:
                            with open(f'{save_path}/{spl_filename[0]}_{key}{spl_filename[1]}', mode='w+b') as f:
                                encoded_img.tofile(f)
                        saved_img_cnt += 1
                except Exception as e:
                    print(e)
                    print(file)

        # 모자른 부분 채우기
        add_aug = num_max - (ratio * file_cnt)
        if num_max > 0 and add_aug > 0:
            random.shuffle(target_list)
            while add_aug > 0:
                file = random.choice(target_list)
                target_key = random.sample(transforms.keys(), 1)[0]
                spl_filename = os.path.splitext(os.path.basename(file))
                if not os.path.exists(f'{save_path}/{spl_filename[0]}_{target_key}{spl_filename[1]}'):
                    img_array = np.fromfile(file, np.uint8)
                    image = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)

                    transformed = transforms[target_key](image=image)
                    transformed_image = transformed["image"]

                    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
                    extension = os.path.splitext(file)[1]  # 이미지 확장자
                    result, encoded_img = cv2.imencode(extension, transformed_image)  # 저장할 확장자로 인코딩
                    add_aug -= 1


#삭제#


if __name__=='__main__':
    DEFAULT_PATH = './merge'
    threshold = 15000
    class_list = ['삭제']

    # 삭제 #

    # 증식
    augment(f'{DEFAULT_PATH}/rename', num_limit=threshold)

    # 데이터셋 분할
    splitfolders.ratio(f'{DEFAULT_PATH}/aug', output=f'{DEFAULT_PATH}/dataset', seed=45, ratio=(.8, 0.2, 0.0))