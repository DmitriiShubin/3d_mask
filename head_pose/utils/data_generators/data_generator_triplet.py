# basic libs
import numpy as np
import json
from scipy import signal
from config import DATA_PATH
from utils.peak_finder import Peak_finder
import random

# pytorch
import torch
from torch.utils.data import Dataset

# custom modules
np.random.seed(42)


class Dataset_train(Dataset):
    def __init__(self, records, aug_on, n_classes, aug_hparams):

        self.n_classes = n_classes
        self.records_list = records
        self.preprocessing = Preprocessing(aug_on=aug_on, aug_hparams=aug_hparams)

        self.triplets = self.create_triplets(1)

    def create_triplets(self, n_mult):

        # get labels for all records

        labels = []
        for record in self.records_list:
            labels.append(json.load(open(DATA_PATH + record[:-4] + '.json'))['label'][1])

        records = np.array(self.records_list)
        labels = np.array(labels)

        triplets = []
        for i in range(n_mult * len(self.records_list)):

            anchor = np.random.choice(self.records_list)
            label_anchor = labels[np.where(records == anchor)]

            pos_subset = records[np.where(labels == label_anchor)]
            pos_subset = pos_subset[pos_subset != anchor]

            neg_subset = records[np.where(labels != label_anchor)]

            pos = np.random.choice(pos_subset.tolist())
            neg = np.random.choice(neg_subset.tolist())

            triplet = {'anchor': anchor, 'pos': pos, 'neg': neg, 'anchor_label': label_anchor}

            triplets.append(triplet)

        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):

        X_anchor, X_pos, X_neg = self.load_data(idx)

        X_anchor = torch.tensor(X_anchor, dtype=torch.float)
        X_pos = torch.tensor(X_pos, dtype=torch.float)
        X_neg = torch.tensor(X_neg, dtype=torch.float)

        return X_anchor, X_pos, X_neg

    def load_data(self, id):

        X_anchor = np.load(DATA_PATH + self.triplets[id]['anchor']).astype(np.float32)
        X_pos = np.load(DATA_PATH + self.triplets[id]['pos']).astype(np.float32)
        X_neg = np.load(DATA_PATH + self.triplets[id]['neg']).astype(np.float32)

        X_anchor = self.preprocessing.run(X=X_anchor, zero_label=not bool(self.triplets[id]['anchor_label']))
        X_pos = self.preprocessing.run(X=X_pos, zero_label=not bool(self.triplets[id]['anchor_label']))
        X_neg = self.preprocessing.run(X=X_neg, zero_label=False)

        return X_anchor, X_pos, X_neg


class Preprocessing:
    def __init__(self, aug_on, aug_hparams=None):

        self.aug_on = aug_on
        self.augmentations = Augmentations(aug_hparams)
        self.peak_finder = Peak_finder()

        # 150Hz Hamming, LPF, 101th order, Fs = 500Hz
        self.weights_LPF = torch.Tensor(
            [
                -5.4919008225336826e-18,
                -0.0005000455211503115,
                0.00032608784631723675,
                0.0003510282248715218,
                -0.0006218245985501345,
                4.9757363256185725e-18,
                0.0007729541114933063,
                -0.0005387763999961042,
                -0.0006100688547703013,
                0.0011198434347491715,
                -3.927340074847161e-18,
                -0.0014409358407893157,
                0.0010081104009800337,
                0.0011387226104913378,
                -0.0020760219259592645,
                -2.406250172789675e-18,
                0.002614412068715733,
                -0.0018058351138413868,
                -0.0020128675626107976,
                0.0036210367932488882,
                -9.30879060644117e-18,
                -0.004444491370527772,
                0.003033464784294073,
                0.0033436719226554172,
                -0.005953360408749405,
                3.707033055210234e-17,
                0.007178404745262026,
                -0.004863507347266427,
                -0.005327399860844995,
                0.00943708736992706,
                -1.5960629281043744e-17,
                -0.011306154943318863,
                0.007651558271959916,
                0.008385132478944539,
                -0.014886192136089904,
                1.896098446056735e-17,
                0.018023735369945715,
                -0.012308236485434174,
                -0.013652634638465121,
                0.02462478061118092,
                -2.134207981263775e-17,
                -0.031239500409895284,
                0.02206461487480888,
                0.02556748938557019,
                -0.04884318393467805,
                2.2870837134127455e-17,
                0.07461628824511687,
                -0.061880494776620894,
                -0.09324379713779846,
                0.3025668539774082,
                0.6002201034788318,
                0.3025668539774082,
                -0.09324379713779844,
                -0.061880494776620894,
                0.07461628824511686,
                2.2870837134127452e-17,
                -0.04884318393467804,
                0.025567489385570182,
                0.022064614874808878,
                -0.03123950040989528,
                -2.134207981263775e-17,
                0.02462478061118091,
                -0.013652634638465118,
                -0.012308236485434174,
                0.018023735369945712,
                1.896098446056735e-17,
                -0.0148861921360899,
                0.008385132478944537,
                0.007651558271959912,
                -0.01130615494331886,
                -1.5960629281043744e-17,
                0.009437087369927056,
                -0.005327399860844994,
                -0.0048635073472664235,
                0.007178404745262023,
                3.7070330552102325e-17,
                -0.005953360408749403,
                0.003343671922655417,
                0.0030334647842940703,
                -0.00444449137052777,
                -9.30879060644117e-18,
                0.003621036793248885,
                -0.0020128675626107976,
                -0.001805835113841385,
                0.002614412068715732,
                -2.406250172789672e-18,
                -0.0020760219259592623,
                0.0011387226104913378,
                0.0010081104009800324,
                -0.0014409358407893149,
                -3.927340074847156e-18,
                0.0011198434347491706,
                -0.0006100688547703013,
                -0.0005387763999961032,
                0.0007729541114933059,
                4.975736325618567e-18,
                -0.0006218245985501341,
                0.0003510282248715218,
                0.00032608784631723653,
                -0.0005000455211503115,
                -5.4919008225336826e-18,
            ]
        )
        self.weights_LPF = self.weights_LPF.view(1, 1, self.weights_LPF.shape[0]).float()
        self.padding_LPF = int((self.weights_LPF.shape[2] - 1) / 2)
        self.padding_LPF = torch.Tensor(np.zeros((self.padding_LPF))).float()

    def run(self, X, zero_label=False):

        # reshape X
        X = np.reshape(X, (1, -1))

        if np.max(X) - np.mean(X) > 10000:
            X = np.clip(X, a_max=np.mean(X) + 300, a_min=0)

        # apply scaling
        if np.std(X) > 0:
            X = (X - np.mean(X)) / np.std(X)
        else:
            X = X - np.mean(X)

        X = X.reshape(-1, 1)

        if self.aug_on:
            X = self.augmentations.run(X, zero_label)

        # X[:, 0] = self.FIR_filt(X[:, 0],weight=self.weights_LPF,padding_vector=self.padding_LPF)

        return X

    def FIR_filt(self, input, weight, padding_vector):
        input = torch.tensor(input).float()
        input = torch.cat((input, padding_vector), 0)
        input = torch.cat((padding_vector, input), 0)
        input = input.view(1, 1, input.shape[0])
        output = torch.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        output = output.view(output.shape[2])
        output = output.detach().numpy()
        output = np.reshape(output, (-1))
        return output


class Augmentations:
    def __init__(self, aug_hparams):

        self.aug_hparams = aug_hparams

    def run(self, X, zero_label):

        if zero_label:
            X = self.make_zeros(X)
            X = self.random_spike(X)

        X = self.gaussian_noise(X)
        X = self.resampling(X)
        X = self.amplitude_adjusting(X)
        X = self.reverse_amplitude(X)
        X[:, 0] = self.baseline_wandering_noise(X[:, 0], fs=500)

        return X

    def make_zeros(self, X):
        if self.coin_flip(self.aug_hparams['make_zeros']['prob']):
            return np.zeros(X.shape)
        else:
            return X

    def reverse_amplitude(self, X):
        if self.coin_flip(self.aug_hparams['reverse_amplitude']['prob']):
            return -1 * X
        else:
            return X

    def baseline_wandering_noise(self, waveform, fs):
        """Adds baseline wandering to the input signal."""

        if self.coin_flip(self.aug_hparams['baseline_wandering_noise']['prob']):

            # Generate time array
            time = np.arange(len(waveform)) * 1 / fs

            # Get number of baseline signals
            baseline_signals = random.randint(1, 5)

            # Loop through baseline signals
            for baseline_signal in range(baseline_signals):
                # Add noise
                waveform += random.uniform(
                    0.01, self.aug_hparams['baseline_wandering_noise']['amp_max']
                ) * np.sin(
                    2
                    * np.pi
                    * random.uniform(0.001, self.aug_hparams['baseline_wandering_noise']['feq_max'])
                    * time
                    + random.uniform(0, 60)
                )

        return waveform

    def gaussian_noise(self, X):

        if self.coin_flip(self.aug_hparams['gaussian_noise']['prob']):

            noise = np.random.normal(
                loc=0, scale=self.aug_hparams['gaussian_noise']['std'], size=X.shape[0]
            ) * np.random.uniform(self.aug_hparams['gaussian_noise']['std'])
            noise = noise.reshape(-1, 1)
            X = X + noise

        return X

    def resampling(self, X):

        x_shape = X.shape[0]

        if self.coin_flip(self.aug_hparams['resampling']['prob']):
            X = signal.resample(
                X,
                num=int(
                    X.shape[0]
                    * (
                        1
                        + 2 * (np.random.uniform() - 0.5) * self.aug_hparams['resampling']['stretching_coef']
                    )
                ),
            )

            if X.shape[0] >= x_shape:
                X = X[:x_shape, :]
            else:
                padding = np.zeros((x_shape - X.shape[0], 1))
                X = np.concatenate([X, padding], axis=0)

        return X

    def random_spike(self, X):

        if self.coin_flip(self.aug_hparams['random_spike']['prob']):
            sample = np.random.choice(np.arange(X.shape[0]), size=int(np.random.uniform(1, 10)))
            X[sample] = 121.45

        return X

    def amplitude_adjusting(self, X):

        if self.coin_flip(self.aug_hparams['amplitude_adjusting']['prob']):
            amp = np.random.uniform(
                1 - self.aug_hparams['amplitude_adjusting']['std'],
                1 + self.aug_hparams['amplitude_adjusting']['std'],
            )
            amp = abs(amp)
            X *= amp

        return X

    @staticmethod
    def coin_flip(probability):
        if random.random() < probability:
            return True
        return False
