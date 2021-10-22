# basic libs
import numpy as np
from tqdm import tqdm
import os
import yaml

# pytorch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

# custom modules
from metrics import Metric
from utils.pytorchtools import EarlyStopping
from utils.post_processing import PostProcessing
from torch.nn.parallel import DataParallel as DP
from utils.data_generators.data_generator import Preprocessing
from time import time

# model
from models.wavenet_triplet.structure import Wavenet


class Model:
    """
    This class handles basic methods for handling the model:
    1. Fit the model
    2. Make predictions
    3. Make inference predictions
    3. Save
    4. Load weights
    5. Restore the model
    6. Restore the model with averaged weights
    """

    def __init__(self, hparams, gpu=None, inference=False):

        self.hparams = hparams
        self.gpu = gpu
        self.inference = inference

        self.start_training = time()

        # ininialize model architecture
        self.__setup_model(inference=inference, gpu=gpu)

        # define model parameters
        self.__setup_model_hparams()

        # declare preprocessing object
        self.preprocessing = Preprocessing(aug_on=False)
        self.postprocessing = PostProcessing()

        self.__seed_everything(42)

    def fit(self, train, valid):

        # setup train and val dataloaders
        train_loader = DataLoader(
            train,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=self.hparams['num_workers'],
        )
        valid_loader = DataLoader(
            valid,
            batch_size=self.hparams['batch_size'],
            shuffle=False,
            num_workers=self.hparams['num_workers'],
        )

        adv_loader = DataLoader(valid, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0)

        # tensorboard
        writer = SummaryWriter(f"runs/{self.hparams['model_name']}_{self.start_training}")

        print('Start training the model')
        for epoch in range(self.hparams['n_epochs']):

            # training mode
            self.model.train()
            avg_loss = 0.0
            adv_loss_running = 0.0

            for X_batch, X_p_batch, X_n_batch in tqdm(train_loader):

                # push the data into the GPU
                X_batch = X_batch.float().to(self.device)
                X_p_batch = X_p_batch.float().to(self.device)
                X_n_batch = X_n_batch.float().to(self.device)

                # clean gradients from the previous step
                self.optimizer.zero_grad()

                # get model predictions
                pred_anchor = self.model(X_batch)
                pred_pos = self.model(X_p_batch)
                pred_neg = self.model(X_n_batch)

                # process main loss
                train_loss = self.loss(pred_anchor, pred_pos, pred_neg)

                # remove data from GPU
                X_batch = X_batch.float().cpu().detach()
                X_p_batch = X_p_batch.float().cpu().detach()
                X_n_batch = X_n_batch.float().cpu().detach()
                pred_anchor = pred_anchor.float().cpu().detach()
                pred_pos = pred_pos.float().cpu().detach()
                pred_neg = pred_neg.float().cpu().detach()

                # calc loss
                avg_loss += train_loss.item() / len(train_loader)

                # backprop
                train_loss.backward()

                # gradient clipping
                if self.apply_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)

                # iptimizer step
                self.optimizer.step()  # self.scaler.step(self.optimizer)
                # self.scaler.update()

            # evaluate the model
            print('Model evaluation')

            # val mode
            self.model.eval()
            avg_val_loss = 0.0

            with torch.no_grad():

                for X_batch, X_p_batch, X_n_batch in tqdm(valid_loader):

                    # push the data into the GPU
                    X_batch = X_batch.float().to(self.device)
                    X_p_batch = X_p_batch.float().to(self.device)
                    X_n_batch = X_n_batch.float().to(self.device)

                    # clean gradients from the previous step
                    self.optimizer.zero_grad()

                    # get model predictions
                    pred_anchor = self.model(X_batch)
                    pred_pos = self.model(X_p_batch)
                    pred_neg = self.model(X_n_batch)

                    avg_val_loss += self.loss(pred_anchor, pred_pos, pred_neg).item() / len(valid_loader)

                    # remove data from GPU
                    X_batch = X_batch.float().cpu().detach()
                    X_p_batch = X_p_batch.float().cpu().detach()
                    X_n_batch = X_n_batch.float().cpu().detach()
                    pred_anchor = pred_anchor.float().cpu().detach()
                    pred_pos = pred_pos.float().cpu().detach()
                    pred_neg = pred_neg.float().cpu().detach()

            # early stopping for scheduler
            if self.hparams['scheduler_name'] == 'ReduceLROnPlateau':
                self.scheduler.step(avg_val_loss)
            else:
                self.scheduler.step()

            es_result = self.early_stopping(score=avg_val_loss, model=self.model, threshold=None)

            # print statistics
            if self.hparams['verbose_train']:
                print(
                    '| Epoch: ',
                    epoch + 1,
                    '| Train_loss: ',
                    avg_loss,
                    '| Val_loss: ',
                    avg_val_loss,
                    '| Current LR: ',
                    self.__get_lr(self.optimizer),
                )

            # add data to tensorboard
            writer.add_scalars(
                'Loss',
                {'Train_loss': avg_loss, 'Val_loss': avg_val_loss},
                epoch,
            )

            # early stopping procesudre
            if es_result == 2:
                print("Early Stopping")
                print(f'global best val_loss model score {self.early_stopping.best_score}')
                break
            elif es_result == 1:
                print(f'save global val_loss model score {avg_val_loss}')

        writer.close()

        # load the best model trained so fat
        self.model = self.early_stopping.load_best_weights()

        return self.start_training

    def predict(self, X_test):
        """
        This function makes:
        1. batch-wise predictions
        2. calculation of the metric for each sample
        3. calculation of the metric for the entire dataset

        Parameters
        ----------
        X_test

        Returns
        -------

        """

        # evaluate the model
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            X_test,
            batch_size=self.hparams['batch_size'],
            shuffle=False,
            num_workers=0,
        )

        error_samplewise = []

        predictions_running = np.empty((0, self.hparams['model']['n_classes']))

        print('Getting predictions')
        with torch.no_grad():
            for X_batch, X_p_batch, X_n_batch in tqdm(test_loader):
                # push the data into the GPU
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                # push the data into the GPU
                X_batch = X_batch.float().to(self.device)
                X_p_batch = X_p_batch.float().to(self.device)
                X_n_batch = X_n_batch.float().to(self.device)

                # clean gradients from the previous step
                self.optimizer.zero_grad()

                # get model predictions
                pred_anchor = self.model(X_batch)
                pred_pos = self.model(X_p_batch)
                pred_neg = self.model(X_n_batch)

                avg_val_loss += self.loss(pred_anchor, pred_pos, pred_neg).item() / len(valid_loader)

                error_samplewise.append([0] * X_batch.shape[0])

        fold_score = self.metric.compute()
        error_samplewise = np.array(error_samplewise)
        predictions_running = np.array(predictions_running)

        self.model = self.early_stopping.load_best_weights()

        return error_samplewise, fold_score, predictions_running

    def predict_inference(self, X):

        X = self.preprocessing.run(X)

        X = X.reshape(1, -1, X.shape[1])

        self.model.eval()
        predictions = self.model.get_embeddings(torch.Tensor(X))
        predictions = predictions.detach().numpy()

        return predictions

    def get_heatmap(self, X):
        X = self.preprocessing.run(X)

        X = X.reshape(1, -1, X.shape[1])

        self.model.eval()
        return self.model.get_cam_activation_maps(torch.Tensor(X)).detach().numpy()

    def save(self, model_path):

        print('Saving the model')

        # states (weights + optimizers)
        if self.gpu != None:
            if len(self.gpu) > 1:
                torch.save(self.model.module.state_dict(), model_path + '.pt')
            else:
                torch.save(self.model.state_dict(), model_path + '.pt')
        else:
            torch.save(self.model.state_dict(), model_path)

        # hparams
        with open(f"{model_path}_hparams.yml", 'w') as file:
            yaml.dump(self.hparams, file)

        return True

    def load(self, model_name):
        self.model.load_state_dict(torch.load(model_name + '.pt', map_location=self.device))
        self.model.eval()
        return True

    @classmethod
    def restore(cls, model_name: str, inference: bool, gpu: list = None):

        if gpu is not None:
            assert all([isinstance(i, int) for i in gpu]), "All gpu indexes should be integer"

        # load hparams
        hparams = yaml.load(open(model_name + "_hparams.yml"), Loader=yaml.FullLoader)

        # construct class
        self = cls(hparams, gpu=gpu, inference=inference)

        # load weights + optimizer state
        self.load(model_name=model_name)

        return self

    @classmethod
    def restore_averaged(cls, models_names: list, gpu: list, inference: bool):

        assert all([isinstance(i, int) for i in gpu]), "All gpu indexes should be int"
        assert all([isinstance(i, str) for i in models_names]), "All models_names should be str"
        assert len(models_names) > 1, "The number of models should be more than 1"

        n_models = float(len(models_names))

        hparams = yaml.load(open(models_names[0] + "_hparams.yml"), Loader=yaml.FullLoader)

        # construct
        # load models
        for count, model_name in enumerate(models_names):
            if count == 0:
                # hparamsclass
                self = cls(hparams, gpu=gpu, inference=inference)

                # load weights + optimizer state
                self.load(model_name=model_name)
                state_dict_main = self.model.state_dict()
                for layer in state_dict_main:
                    state_dict_main[layer] = state_dict_main[layer] / n_models

            else:
                # hparams
                hparams = yaml.load(open(model_name + ".yml"), Loader=yaml.FullLoader)

                # construct class
                model_add = cls(hparams, gpu=gpu, inference=inference)

                # load weights + optimizer state
                model_add.load(model_name=model_name)
                state_dict_add = model_add.model.state_dict()
                for layer in state_dict_main:
                    state_dict_main[layer] = state_dict_main[layer] + (state_dict_add[layer]) / n_models

            self.model.load_state_dict(state_dict_main)

        return self

    ################## Utils #####################

    def __get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def __setup_model(self, inference, gpu):

        # TODO: re-write to pure DDP
        if inference or gpu is None:
            self.device = torch.device('cpu')
            self.model = Wavenet(hparams=self.hparams['model']).to(self.device)
        else:
            if torch.cuda.device_count() > 1:
                if len(gpu) > 1:
                    print("Number of GPUs will be used: ", len(gpu))
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = Wavenet(hparams=self.hparams['model']).to(self.device)
                    self.model = DP(self.model, device_ids=gpu, output_device=gpu[0])
                else:
                    print("Only one GPU will be used")
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = Wavenet(hparams=self.hparams['model']).to(self.device)
            else:
                self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                self.model = Wavenet(hparams=self.hparams['model']).to(self.device)
                print('Only one GPU is available')

        print('Cuda available: ', torch.cuda.is_available())

        # prune model
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                prune.l1_unstructured(
                    module, name='weight', amount=self.hparams['model']['pruning_factor_cnn']
                )

            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(
                    module, name='weight', amount=self.hparams['model']['pruning_factor_dnn']
                )

        return True

    def __setup_model_hparams(self):

        # 1. define losses
        # weights = torch.Tensor([1.54133065, 1.0, 1.03801765]).to(self.device)
        self.loss = nn.TripletMarginLoss(margin=10.0, p=2)

        # 2. define model metric
        self.metric = Metric(self.hparams['model']['n_classes'])

        # 3. define optimizer
        self.optimizer = eval(f"torch.optim.{self.hparams['optimizer_name']}")(
            params=self.model.parameters(), **self.hparams['optimizer_hparams']
        )

        # 4. define scheduler
        self.scheduler = eval(f"torch.optim.lr_scheduler.{self.hparams['scheduler_name']}")(
            optimizer=self.optimizer, **self.hparams['scheduler_hparams']
        )

        # 5. define early stopping
        self.early_stopping = EarlyStopping(
            checkpoint_path=self.hparams['checkpoint_path']
            + f'/checkpoint_{self.start_training}'
            + str(self.hparams['start_fold'])
            + '.pt',
            patience=self.hparams['patience'],
            delta=self.hparams['min_delta'],
            is_maximize=False,
        )

        # 6. set gradient clipping
        self.apply_clipping = self.hparams['clipping']  # clipping of gradients

        # 7. Set scaler for optimizer
        self.scaler = torch.cuda.amp.GradScaler()

        return True

    def __seed_everything(self, seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
