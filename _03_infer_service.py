import os

import pandas as pd
import torch
import requests
from io import BytesIO
from datetime import datetime, timedelta
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms

from utils.eval_utils import EvalResult
from utils.common_utils import Params, MyModel, DBCommon
import math
import time



# Inferecne, Main Class
class Inference(DBCommon):
    def __init__(self):
        start = time.time()
        math.factorial(1234567)

        self.params = Params(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
        self.params.device = self.device
        print(self.params.device, type(self.params.device))
        self.er = EvalResult(self.params)
        self.model = self.get_model(artifact_uri='')
        self.now = datetime.now().strftime('%y%m%d_%H%M%S')

        sec = (time.time() - start)
        result_list = str(timedelta(seconds=sec)).split(".")
        print(result_list[0])

    def get_rows(self):
        try:
            conn = self.connect()
            sql_str = """
                삭제
                """

            df = self.select_df(conn, sql_str)
            print(df)
        finally:
            conn.close()
        return df

    def load_img_from_db(self, image_path):
        prefix = '삭제'
        img_url = f'{prefix}/{image_path}'
        print(img_url)
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')

        return img

    def get_model(self, artifact_uri=None):
        mm = MyModel(self.params)
        model = mm.get_model(model_name=self.params.model_name, num_classes=self.params.num_classes, eval=True,
                             artifact_uri=artifact_uri)
        model = model.to(self.device)
        return model

    def infer_one(self, img_path):
        img = self.load_img_from_db(img_path)
        transform = transforms.Compose([transforms.Resize((self.params.input_size, self.params.input_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=self.params.mean, std=self.params.std),
                                        ])
        x = transform(img).unsqueeze(0) # .unsqueeze(0) <- batch size dimension
        x = x.to(self.device)
        # print(x.shape)

        output = self.model(x)

        # confidence score 산출을 위한 softmax 추가
        output = torch.nn.functional.softmax(output, dim=1)
        confidence_score, y_pred = torch.max(output, 1)
        confidence_score = round(float(confidence_score), 3)
        pred_idx = int(y_pred[0].cpu().numpy())
        pred_label = self.params.class_list[pred_idx]

        return img, pred_idx, pred_label, confidence_score

    def visualized(self, img, pred_label, file_name):
        # display the PIL image
        draw = ImageDraw.Draw(img, 'RGBA')
        fontpath = "font/malgun.ttf"
        font_size = round(10* (img.size[0]/224))
        font = ImageFont.truetype(fontpath, font_size)
        outline_rgba = (111, 251, 174, 255)
        fill_rgba = (111, 251, 174, 180)

        x, y = 10, 10
        w, h = font.getsize(pred_label) # font에 따라 text width, height 구하기

        # text 영역의 뒷배경
        draw.rectangle((x, y, x + w, y + h),  outline=outline_rgba, fill=fill_rgba)
        draw.text((x, y), pred_label, font=font, fill=(0, 0, 0, 255))

        file_name = '-'.join(file_name.split('/'))
        img.save(os.path.join(self.save_path, file_name))


    def insert_result_tp_list(self, result_list):
        try:
            sql_str = "삭제"
            conn = self.connect()
            self.insert_list(conn, sql_str, result_list)
        finally:
            conn.close()


    def main(self):
        df = self.get_rows()
        self.save_path = os.path.join('result')
        os.makedirs(self.save_path, exist_ok=True)

        result_list = []
        for idx, row in df.iterrows():
            img, pred_idx, pred_label, confidence_score = self.infer_one(row['IMG_FILE'])
            self.visualized(img, pred_label, row['IMG_FILE'])
            print('----------------------------------------')
            result_list.append(('삭제'))
        self.insert_result_tp_list(result_list)

if __name__ == '__main__':
    config_path = '삭제'
    inf = Inference()
    inf.main()