import yaml
import torch
import mlflow
import snowflake.connector

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


class MyModel:
    def __init__(self, params):
        self.model = None
        self.criterion = None
        self.optimizer = None

        self.params = params
        self.model_name = params.model_name
        self.num_classes = len(params.class_list)
        self.loss_fn = params.loss_fn
        self.optm_fn = params.optm_fn
        self.lr_scheduler = params.lr_scheduler
        self.eval = False

    def get_model(self, model_name, num_classes, eval=False, artifact_uri=None):
        if 0 <= model_name.find('efficient'):

            from efficientnet_pytorch import EfficientNet
            if eval:
                # just loaded model's structure
                self.model = EfficientNet.from_name(f'efficientnet-b{self.params.coef}', num_classes=num_classes)
                self._set_eval_model(model_name, artifact_uri)
                self.model.requires_grad_(False)
            else:
               # self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                print(f'model_name={model_name}')
                self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)

        elif 0 <= model_name.find('densenet'):
            from models.DenseNet import DenseNet121
            self.model = DenseNet121(num_classes=self.num_classes)
            if eval:
                self._set_eval_model(model_name, artifact_uri)
                self.model.requires_grad_(False)

        elif 0 <= model_name.find('resnet'):

            from models.ResNet import resnet50
            self.model = resnet50(num_classes=self.num_classes)
            if eval:
                self._set_eval_model(model_name, artifact_uri)
                self.model.requires_grad_(False)

        elif 0 <= model_name.find('wd_resnet'):
            import models.WideResNet as WideResNet
            # depth=28, num_classes=3, widen_factor=10, dropRate=0.0
            self.model = WideResNet(self.params.layers, self.num_classes, widen_factor=self.params.widen_factor, dropRate=self.params.droprate)
            if eval:
                self._set_eval_model(model_name, artifact_uri)
                self.model.requires_grad_(False)

        elif 0 <= model_name.find('vggnet'):
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
            self.model.requires_grad_(False)
            num_features = self.model.classifier[6].in_features
            # Remove last layer
            features = list(self.model.classifier.children())[:-1]
            # fit our model's output size
            features.extend([torch.nn.Linear(num_features, self.num_classes)])
            # Replace the model classifier
            self.model.classifier = torch.nn.Sequential(*features)

            if eval:
                self._set_eval_model(model_name, artifact_uri)

        return self.model

    def _set_eval_model(self, model_name, artifact_uri):
        if artifact_uri is not None:
            state_dict = mlflow.pytorch.load_state_dict(artifact_uri, map_location=self.params.device)
            # self.model.load_state_dict(torch.load(state_dict['model']))
            self.model.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(torch.load(model_name, map_location=self.params.device))
        self.model.eval()

    def get_loss_fn(self):
        if self.loss_fn == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.loss_fn == 'focal_loss':
            self.criterion = None
        return self.criterion

    def get_optm_fn(self):
        # lr 3e-4
        if self.optm_fn == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9, weight_decay=self.params.weight_decay)
        elif self.optm_fn == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        elif self.optm_fn == 'sgdp':
            from adamp import SGDP
            self.optimizer = SGDP(self.model.parameters(), lr=self.params.lr, momentum=0.9, weight_decay=self.params.weight_decay)
        return self.optimizer

    def get_lr_scheduler(self):
        if self.lr_scheduler == 'mp':
            lmbda = lambda epoch: 0.98739
            lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lmbda)
        elif self.lr_scheduler == 'ms':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        elif self.lr_scheduler == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        elif self.lr_scheduler == 'cosine_wr':
            # optimizer momentum, wide resnet
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
        elif self.lr_scheduler == 'cosine_an':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        return lr_scheduler



class CustomMlflow:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        mlflow.set_tracking_uri("http://localhost:500")
        tracking_uri = mlflow.get_tracking_uri()
        print(tracking_uri)

    def make_experiments(self, exp_name):
        experiment_id = mlflow.create_experiment(exp_name)
        experiment = mlflow.get_experiment(experiment_id)
        print("Name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
        return experiment_id

    def mlflow_setting(self):
        experiment = mlflow.get_experiment_by_name(self.exp_name)
        print(experiment)
        if experiment == None:
            experiment = self.make_experiments(self.exp_name)
        mlflow.set_experiment(self.exp_name)
       # print('END set_experiment()')
       # self.get_experiment_info(experiment)

        return mlflow

    def print_auto_logged_info(self, r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in mlflow.tracking.MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        # print("tags: {}".format(tags))

    def get_experiment_info(self, experiment):
        print("Name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


# DB Connector 관련 클래스
class DBCommon:
    def connect(self):
        conn = snowflake.connector.connect(
            user='삭제'
        )
        return conn

    def select_list(self, conn, sql_str):
        try:
            cursor = conn.cursor()
            cursor.execute(sql_str)
            rows = cursor.fetchall()
            # print(rows)
        finally:
            cursor.close()
        return rows

    def select_df(self, conn, sql_str):
        try:
            cursor = conn.cursor()
            cursor.execute(sql_str)
            rows = cursor.fetch_pandas_all()
            # print(rows)
        finally:
            cursor.close()
        return rows

    def insert_list(self, conn, sql_str, rows_to_insert):
        result_cnt = -1
        print('debug')
        try:
            cursor = conn.cursor()
            cursor.executemany(sql_str, rows_to_insert)
            result_cnt = cursor.rowcount
            print(result_cnt)
            print(len(rows_to_insert))

            if len(rows_to_insert) == result_cnt:
                cursor.execute("commit")
            else:
                cursor.execute("rollback")
                cursor.execute
                raise Exception('Insert Error')
        except Exception as e:
            print(f'[{e}] row count를 확인하세요 >> len(rows_to_insert)={len(rows_to_insert)}, result_cnt={result_cnt}')
        finally:
            cursor.close()


