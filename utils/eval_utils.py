import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



class EvalResult:
    def __init__(self, params):
        self.class_label = params.class_list
        self.now = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    def imshow(self, inp, title=None):
        # tensor -> image -> plt
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([ 0.485, 0.456, 0.406 ])
        std = np.array([ 0.229, 0.224, 0.225 ])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.show()
        plt.savefig('result/eval_result.png')
        # plt.pause(0.001)  # pause a bit so that plots are updated

    def image_level(self, y_test, y_predict):
        # Image Level F1 Score
        # binary classification; return best score in f1 score by all classes each other
        # print("Image Level F1 Score: {:.4f}".format(max(f1_score(y_test, y_predict, pos_label=1, average=None))))
        f1score_each_classes = f1_score(y_test, y_predict, pos_label=1, average=None)

        total_f1_score_txt = "Total F1 Score: {:.4f}".format(f1_score(y_test, y_predict, average='macro'))
        save_txt = '\n'.join(['{} F1 Score: {:.4f}'.format(self.class_label[idx], one_score) for idx, one_score in
                              enumerate(f1score_each_classes.tolist())])
        save_txt = f'{save_txt}\n\n{total_f1_score_txt}'
        print(save_txt)
        return save_txt


    def report_roc(self, y_test, y_predict, y_predict_proba):
        print(classification_report(y_test, y_predict, target_names=self.class_label))
        # ROC curve
        # y_predict_proba[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_predict_proba)
        roc_auc = metrics.auc(fpr, tpr)
        print("ROC curve fpr: {}".format(','.join([str(i) for i in fpr])))
        print("ROC curve tpr: {}".format(','.join([str(i) for i in tpr])))
        print("ROC curve auc: {}".format(str(roc_auc)))
        self._draw(fpr, tpr, roc_auc)


    def _draw(self, fpr, tpr, roc_auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Breast Canner Classification ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("result/roc-{}.png".format(self.now))


    def plot_confusion_matrix(self, target_total, prediction_total, labels, title):
        plt.figure()
        cm = confusion_matrix(target_total, prediction_total)
        print(cm)
        cm_1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 정규화
        print(cm_1)

        plt.imshow(cm_1, interpolation='nearest')  # 특정 창에 이미지 표시
        plt.title('Confusion Matrix')
        plt.colorbar()

        num_local = np.array(range(len(labels)))
        plt.xticks(num_local, labels)  # x축 좌표에 레이블
        plt.yticks(num_local, labels)  # y축 좌표에 레이블
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # norm, float 연산 시 부동소수점 오차 발생을 방지하기 위해 round, f-string 적용
        for i in range(cm_1.shape[0]):
            for j in range(cm_1.shape[1]):
                plt.text(j, i, f'{round(cm_1[i][j]* 100, 2):.02f}\n({cm[i][j]})', ha="center", va="center", color="white")

        # count
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         plt.text(j, i, cm[i][j], ha="center", va="center", color="white")
        save_path = f'result/{self.now}_{title}_matrix.png'
        plt.savefig(save_path)

        return save_path

