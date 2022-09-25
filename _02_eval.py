import os
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from utils.eval_utils import EvalResult
from utils.common_utils import Params, MyModel
from tqdm import tqdm
import numpy as np
import shutil
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw


class EvalModel:
    def __init__(self, config_path):
        self.params = Params(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
        self.er = EvalResult(self.params)

    def get_dataloader(self):
        transform = transforms.Compose([transforms.Resize((self.params.input_size, self.params.input_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=self.params.mean, std=self.params.std),
                                        ])
        test_set = datasets.ImageFolder(root=f'{self.params.data_root}/{self.params.test_set}', transform=transform)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

        return test_dataloader

    def check_pred_result(self, all_files, target_total, prediction_total, save_img=True):
        check_list = []
        now = datetime.now().strftime('%y%m%d_%H%M%S')

        for file_path, target, pred in tqdm(zip(all_files, target_total, prediction_total)):
            check_list.append(f'{os.path.basename(file_path)}\t{file_path}\t{self.params.class_list[target]}\t{self.params.class_list[pred]}\t{bool(target == pred)}')
            if save_img and (target != pred):
                save_path = os.path.join(r'Z:\05_모델추론결과', now, self.params.class_list[pred])
                os.makedirs(save_path, exist_ok=True)
                file_name = f'{self.params.class_list[target]}-{os.path.basename(file_path)}'
                shutil.copy(file_path, os.path.join(save_path, file_name))
        check_list_txt = '\n'.join(check_list)
        check_list_txt = f'file_name\tfile_path\ttarget\tpred\tT/F\n{check_list_txt}'
        return check_list_txt

    def test_and_visualize_model(self, model, criterion, test_dataloader, num_images=4, save_img=False):
        running_loss, running_corrects, num_cnt = 0.0, 0, 0
        target_total = []
        prediction_total = []
        output_total = []

        all_files, _ = map(list, zip(*test_dataloader.dataset.samples))

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)  # batch의 평균 loss 출력

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += inputs.size(0)  # batch size

                # 기록용
                target = labels.data.cpu().numpy().tolist()
                prediction = preds.cpu().numpy().tolist()

                target_total.extend(target)
                prediction_total.extend(prediction)
                output_total.extend([prediction.count(index) / len(prediction) for index in range(len(self.params.class_list))])

                # 예시 그림 plot
                if i < num_images :
                    ax = plt.subplot(1, 1, 1)
                    ax.axis('off')
                    ax.set_title('%s : %s -> %s' % (
                        'True' if self.params.class_list[labels[0].cpu().numpy()] == self.params.class_list[
                            preds[0].cpu().numpy()]
                        else 'False', self.params.class_list[labels[0].cpu().numpy()],
                        self.params.class_list[preds[0].cpu().numpy()]))
                    self.er.imshow(inputs.cpu().data[0])

            test_loss = running_loss / num_cnt
            test_acc = running_corrects.double() / num_cnt
            print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc * 100))

            f1_score_txt = self.er.image_level(target_total, prediction_total)
            # self.er.report_roc(target_total, prediction_total, np.array(output_total))
            cm_save_path = self.er.plot_confusion_matrix(target_total, prediction_total, self.params.class_list, self.params.model_name)
            check_list_txt = self.check_pred_result(all_files, target_total, prediction_total, save_img=save_img)

            # print(check_list_txt)

        return [cm_save_path, f1_score_txt, check_list_txt]

    def run(self, model_path):
        mm = MyModel(self.params)
        model = mm.get_model(model_name=model_path, num_classes=self.params.num_classes, eval=True)
        model = model.to(self.device)
        criterion = mm.get_loss_fn()

        test_dataloader = self.get_dataloader()

        _ = self.test_and_visualize_model(model, criterion, test_dataloader, num_images=0)

    def run_with_mlflow(self, artifact_uri):
        from utils.common_utils import CustomMlflow
        custom_mlflow = CustomMlflow('photo_review_classification-local')
        mlflow = custom_mlflow.mlflow_setting()
        with mlflow.start_run(run_name=f'{self.params.model_name}-eval') as run:
            print('>>>>>> start mlflow for eval')
            mlflow.log_param('model_uri', artifact_uri)
            mm = MyModel(self.params)
            model = mm.get_model(model_name=self.params.model_name, num_classes=self.params.num_classes, eval=True, artifact_uri=artifact_uri)
            model = model.to(self.device)
            criterion = mm.get_loss_fn()

            test_dataloader = self.get_dataloader()

            cm_save_path, f1_score_txt, check_list_txt = self.test_and_visualize_model(model, criterion, test_dataloader, num_images=0, save_img=True)

            mlflow.log_artifact(cm_save_path, 'result')
            mlflow.log_text(f1_score_txt, 'result/f1score.txt')
            mlflow.log_text(check_list_txt, 'result/check_list.txt')


if __name__=='__main__':

    config_path = './config/efficientnet_3.yml'
    em = EvalModel(config_path)
    artifact_uri=''
    em.run_with_mlflow(artifact_uri)