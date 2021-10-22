# import
import numpy as np
import json
import pandas as pd
import torch
import os
from config import SPLIT_TABLE_PATH, DEBUG_FOLDER


from metrics import Metric


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class CVPipeline:
    def __init__(self, hparams, gpu, model, Dataset_train):

        # load the model

        self.hparams = hparams
        self.gpu = gpu
        self.Dataset_train = Dataset_train

        print('\n')
        print('Selected Learning rate:', self.hparams['optimizer_hparams']['lr'])
        print('\n')

        self.exclusions = []

        self.splits, self.splits_test = self.load_split_table()
        self.metric = Metric(self.hparams['model']['n_classes'])

        self.model = model

    def load_split_table(self):

        splits = []
        split_files = [
            i for i in os.listdir(SPLIT_TABLE_PATH) if i.find('table.json') != -1 and i.find('test') == -1
        ]

        for i in split_files:
            data = json.load(open(SPLIT_TABLE_PATH + i))

            splits.append(data)

        splits_cv = pd.DataFrame(splits)

        splits = []
        split_files = [i for i in os.listdir(SPLIT_TABLE_PATH) if i.find('test') != -1]

        for i in split_files:
            data = json.load(open(SPLIT_TABLE_PATH + i))

            splits.append(data)

        splits_test = pd.DataFrame(splits)

        return splits_cv, splits_test

    def train(self):

        start_training = []
        fold_scores_val = []
        fold_scores_test = []

        self.model = self.model(hparams=self.hparams, gpu=self.gpu)

        for fold in range(self.splits.shape[0]):

            if fold is not None:
                if fold not in self.hparams['start_fold']:
                    continue

            train = self.Dataset_train(
                self.splits['train'].values[fold],
                aug_on=True,
                n_classes=self.hparams['model']['n_classes'],
                aug_hparams=self.hparams['model']['aug_hparams'],
            )
            valid = self.Dataset_train(
                self.splits['val'].values[fold],
                aug_on=False,
                n_classes=self.hparams['model']['n_classes'],
                aug_hparams=self.hparams['model']['aug_hparams'],
            )
            test = self.Dataset_train(
                self.splits_test['test'].values[0],
                aug_on=False,
                n_classes=self.hparams['model']['n_classes'],
                aug_hparams=self.hparams['model']['aug_hparams'],
            )

            # train model
            start_training.append(self.model.fit(train=train, valid=valid))

            # get model predictions
            error_val, fold_score, pred_val = self.model.predict(valid)
            error_test, fold_score_test, pred_test = self.model.predict(test)

            print("Model's final scrore, cv: ", fold_score)
            print("Model's final scrore, test: ", fold_score_test)

            fold_scores_val.append(fold_score)
            fold_scores_test.append(fold_score_test)

            # save the model
            self.model.save(
                self.hparams['model_path']
                + '/'
                + self.hparams['model_name']
                + f"_{fold}"
                + '_fold_'
                + str(np.round(fold_score, 2))
                + '_'
                + str(np.round(fold_score_test, 2))
                + '_'
                + str(start_training[-1])
            )

            # save data for debug
            # self.save_debug_data(error_val, pred_val, self.splits['val'].values[fold])
            self.save_debug_data(error_test, pred_test, self.splits_test['test'].values[0])

        return fold_scores_val, fold_scores_test, start_training

    def save_debug_data(self, error, pred, validation_list):

        for index, data in enumerate(validation_list):

            patient_fold = data.split('/')[-2]
            data = data.split('/')[-1]

            out_json = {}
            out_json['prediction'] = pred[index].tolist()
            out_json['error'] = error[index].tolist()

            os.makedirs(DEBUG_FOLDER + patient_fold, exist_ok=True)
            # save debug data
            with open(DEBUG_FOLDER + patient_fold + '/' + f'{data[:-4]}.json', 'w') as outfile:
                json.dump(out_json, outfile)

        return True
