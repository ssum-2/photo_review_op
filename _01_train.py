import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from torchmetrics import F1Score
import matplotlib.pyplot as plt
import time, datetime
import copy
import random
from torch.utils.data import Subset
from utils.common_utils import Params, MyModel, CustomMlflow
import argparse
from tqdm import tqdm
import os
from _02_eval import EvalModel

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['AWS_PROFILE'] = 'mlflow' # server aws credential option

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_dataset():
    ## make dataset
    transform = transforms.Compose([transforms.Resize((params.input_size, params.input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=params.mean, std=params.std),
                                    ])
    train_set = datasets.ImageFolder(root=f'{params.data_root}/{params.train_set}', transform=transform)
    val_set = datasets.ImageFolder(root=f'{params.data_root}/{params.val_set}', transform=transform)
    test_set = datasets.ImageFolder(root=f'{params.data_root}/{params.test_set}', transform=transform)

    print(f'train_set={len(train_set)} | val_set={len(val_set)} | test_set={len(test_set)}')

    # data loader
    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=4)
    dataloaders['valid'] = torch.utils.data.DataLoader(val_set, batch_size=params.batch_size, shuffle=False, num_workers=4)
    dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=4)

    batch_num['train'], batch_num['valid'], batch_num['test'] = len(train_set), len(val_set), len(test_set)
    print('batch_size : %d,  tvt : %d / %d / %d' % (params.batch_size, batch_num['train'], batch_num['valid'], batch_num['test']))

    return dataloaders, batch_num


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(params.mean)
    std = np.array(params.std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def check_img_dataset(dataloaders):
    num_show_img = 5

    # train check
    inputs, classes = next(iter(dataloaders['train']))
    out = make_grid(inputs[:num_show_img])
    imshow(out, title=[params.class_list[int(x)] for x in classes[:num_show_img]])
    # valid check
    inputs, classes = next(iter(dataloaders['valid']))
    out = make_grid(inputs[:num_show_img])
    imshow(out, title=[params.class_list[int(x)] for x in classes[:num_show_img]])
    # test check
    inputs, classes = next(iter(dataloaders['test']))
    out = make_grid(inputs[:num_show_img])
    imshow(out, title=[params.class_list[int(x)] for x in classes[:num_show_img]])


def train_model(model, device, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # mlflow log setting
    mlflow.log_param('start', since)
    mlflow.log_param('model_name', params.model_name)
    mlflow.log_param('num_classes', len(params.class_list))
    mlflow.log_param('data_path', params.data_root)
    mlflow.log_param('lr', params.lr)
    mlflow.log_param('lr_scheduler', params.lr_scheduler)
    mlflow.log_param('loss_fn', params.loss_fn)
    mlflow.log_param('optm_fn', params.optm_fn)
    mlflow.log_param('batch_size', params.batch_size)
    mlflow.log_param('num_epochs', params.num_epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    f1 = F1Score(num_classes=len(params.class_list)).to(device)
    acc_threshold = 80.0
    early_stop = False

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            losses = AverageMeter()
            top1 = AverageMeter()
            f1_score = AverageMeter()

            with tqdm(dataloaders[phase], unit='batch') as tepoch:
                # Iterate over data.
                # for inputs, labels in tqdm(dataloaders[phase]):
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    # inputs.size(0) == batch_size
                    running_loss += loss.item() * inputs.size(0)    # losses.sum
                    running_corrects += torch.sum(preds == labels.data)
                    num_cnt += len(labels)  # len(dataset)

                    # 1 iteration loss, acc, f1score 계산; 3320 / 16 / 20 = 1epoch 당 210 iteration
                    prec1 = accuracy(outputs.data, labels, topk=(1,))[0]
                    losses.update(loss.data.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    f1_score.update(f1(preds, labels).item(), inputs.size(0))

                    # set values progress bar
                    tepoch.set_postfix(phase=phase, f1_score_per_iter=f1_score.val, top1_prec=prec1.item()
                                       , loss_per_iter=losses.val, acc=float(torch.sum(preds == labels.data) / len(labels)), loss=losses.avg)
                    time.sleep(0.1)

                    # mlflow tracking
                    mlflow.log_metric(key='Loss per iter', value=losses.val, step=tepoch.n)
                    mlflow.log_metric(key='Top1 Prec per iter', value=top1.val, step=tepoch.n)
                    mlflow.log_metric(key='Acc per iter', value=float(torch.sum(preds == labels.data) / len(labels)), step=tepoch.n)
                    mlflow.log_metric(key='F1 score per iter', value=f1_score.val, step=tepoch.n)

            ### 삭제 ###

                # early stopping
                elif (epoch-best_idx) <= params.patience:
                    early_stop = True
                    break
                # model save
                elif epoch % 10 == 0: # acc_threshold <= top1.avg or
                    torch.save(model.state_dict(), f'checkpoint/{now}_{args.config}_acc-{top1.avg:.2f}_loss-{losses.avg:.2f}_ep-{epoch}.pth')

                    # mlflow log model state dict
                    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "loss": loss}
                    art_path = f'checkpoint/{now}_{args.config}_acc-{top1.avg:.2f}_ep-{epoch}'
                    mlflow.pytorch.log_state_dict(state_dict, artifact_path=art_path)
        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    #삭제#

    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": best_idx, "loss": valid_loss[best_idx]}
    mlflow.pytorch.log_state_dict(state_dict, artifact_path='result/best')

    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc



def plot_result(best_idx, train_acc, train_loss, valid_acc, valid_loss):
    print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-')
    ax1.plot(valid_acc, 'r-')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-')
    ax2.plot(valid_loss, 'k-')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    # plt.show()
    # plt.savefig(f'result/{now}_{model_name}_plot_acc-{valid_acc[best_idx]}_ep-{best_idx}.png')
    mlflow.log_figure(fig, f'result/{now}_{args.config}_plot_acc-{valid_acc[best_idx]}_ep-{best_idx}.png')


def main():
    # 기본 설정
    random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    # params.input_size = EfficientNet.get_image_size(params.model_name)

    # 데이터셋 생성
    dataloaders, batch_num = get_dataset()

    # 데이터 확인
    # check_img_dataset(dataloaders, params)

    # 모델 불러오기
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
    print(f'>>> set device={device}')
    mm = MyModel(params)
    model = mm.get_model(params.model_name, len(params.class_list), eval=False)
    model = model.to(device)

    # loss function
    criterion = mm.get_loss_fn()

    # optmizer
    optimizer_ft = mm.get_optm_fn()

    # learning rate scheduler
    exp_lr_scheduler = mm.get_lr_scheduler()

    # train model
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, device, dataloaders, criterion,
                                                                                          optimizer_ft,
                                                                                          exp_lr_scheduler,
                                                                                          num_epochs=params.num_epochs)
    # model result 확인
    plot_result(best_idx, train_acc, train_loss, valid_acc, valid_loss)

    # model evaluate 결과 확인 및 mlflow에 저장
    em = EvalModel(f'./config/{args.config}.yml')
    test_dataloader = em.get_dataloader()
    model.eval()
    eval_result = em.test_and_visualize_model(model, criterion, test_dataloader, num_images=0)
    # eval_result = [cm_save_path, f1_score_txt, check_list_txt]
    mlflow.log_artifact(eval_result[0], 'result')


if __name__=='__main__':
    now = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', type=str, required=False)
    args = parser.parse_args()

    params = Params(f'{args.config}')

    custom_mlflow = CustomMlflow('photo_review_classification')
    mlflow = custom_mlflow.mlflow_setting()

    with mlflow.start_run(run_name=args.config) as run:
        main()

    custom_mlflow.print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))






