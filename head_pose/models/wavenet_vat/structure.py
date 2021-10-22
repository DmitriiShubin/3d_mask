import torch
import torch.nn as nn
from torch.autograd import Function
from models.wavenet_pre_trained.structure import WaveNet as Wavenet_pre_trained
import yaml
import numpy as np


class CBRD_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, drop_rate, pool_size, dilation):
        super().__init__()

        self.conv = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=pool_size)
        self.drop = nn.Dropout(drop_rate)

        self.conv2 = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            stride=1,
            bias=False,
        )

        self.conv3 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            stride=1,
            bias=False,
        )

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.relu(x)
        x += identity
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RevGrad_func(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


class RevGrad(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        self.revgrad = RevGrad_func.apply

        super().__init__(*args, **kwargs)

    def forward(self, input_):
        return self.revgrad(input_)


class Wavenet(Wavenet_pre_trained):
    def __init__(
        self,
        hparams,
    ):
        hparams_pre_trained = yaml.load(
            open(hparams['pre_trained_model'] + "_hparams.yml"), Loader=yaml.FullLoader
        )
        hparams_pre_trained = hparams_pre_trained['model']
        super().__init__(hparams=hparams_pre_trained)

        self.load_state_dict(torch.load(hparams['pre_trained_model'] + '.pt'))

        self.hparams = hparams

        # probing to calculate the number of layers
        dummy = torch.Tensor([0] * hparams['n_samples']).view(1, 1, -1)
        dummy = self.stem_encoder(dummy)
        skip_outputs = []
        for layer in self.wavenet_encoder:
            dummy, skip_out = layer(dummy)
            skip_outputs.append(skip_out)

        result = torch.mean(torch.stack(skip_outputs), dim=0)

        out_shape = result.shape[-2]

        self.out_layer_1 = nn.Linear(out_shape, out_shape)
        self.out_layer_2 = nn.Linear(out_shape, hparams['n_classes'])

        self.revgrad_1_1 = RevGrad()
        self.revgrad_1_2 = RevGrad()
        self.revgrad_1_3 = RevGrad()
        self.revgrad_1_4 = RevGrad()
        self.revgrad_1_5 = RevGrad()
        self.revgrad_1_6 = RevGrad()
        self.revgrad_1_7 = RevGrad()
        self.revgrad_1_8 = RevGrad()
        self.revgrad_1_9 = RevGrad()
        self.revgrad_1_10 = RevGrad()

        self.revgrad_2_1 = RevGrad()
        self.revgrad_2_2 = RevGrad()
        self.revgrad_2_3 = RevGrad()
        self.revgrad_2_4 = RevGrad()
        self.revgrad_2_5 = RevGrad()
        self.revgrad_2_6 = RevGrad()
        self.revgrad_2_7 = RevGrad()
        self.revgrad_2_8 = RevGrad()
        self.revgrad_2_9 = RevGrad()
        self.revgrad_2_10 = RevGrad()

        self.out_s_layer_1 = nn.Linear(
            len(hparams['wavenet_dilation']) * hparams['layer_feature_maps'][-1] * 4,
            len(hparams['wavenet_dilation']) * hparams['layer_feature_maps'][-1] * 4,
        )
        self.out_s_layer_2 = nn.Linear(
            len(hparams['wavenet_dilation']) * hparams['layer_feature_maps'][-1] * 4, 1
        )

        # self.out_s_layer_1 = nn.Linear(
        #     2 * hparams['layer_feature_maps'][-1], 2 * hparams['layer_feature_maps'][-1],
        # )
        # self.out_s_layer_2 = nn.Linear(2 * hparams['layer_feature_maps'][-1], 1)

        # freeze encoder
        if hparams['freeze_layers']:
            for param in self.stem_encoder.parameters():
                param.requires_grad = False
            for param in self.wavenet_encoder.parameters():
                param.requires_grad = False

    def forward(self, x, x_s=None, train=False):

        x = x.permute(0, 2, 1)

        if train:

            x_s = x_s.permute(0, 2, 1)
            out, feature_maps = self.main_head(x)
            adv_out = self.adv_head(x_s, feature_maps)

            return out, adv_out

        else:
            out, _ = self.main_head(x)
            return out

    def get_cam_activation_maps(self, x):

        x = x.permute(0, 2, 1)

        shape = x.shape[-1]

        _, x = self.__encoder_block(x)

        x = torch.sum(torch.abs(x[0]), dim=1)

        x_up = torch.Tensor(np.zeros(shape))

        scale = int(shape / x.shape[-1])

        for i in range(x.shape[-1]):
            x_up[i * scale : (i + 1) * scale] = x[0, i]

        return x_up

    def main_head(self, x):

        x, feature_maps = self.__encoder_block(x)

        x = torch.mean(x, dim=2)

        x = torch.relu(self.out_layer_1(x))
        x = self.out_layer_2(x) / self.hparams['temperature']
        x = torch.softmax(x, dim=1)
        return x, feature_maps

    def adv_head(self, x_s, feature_maps):

        _, feature_maps_s = self.__encoder_block(x_s)

        # x_mean_s_1 = self.revgrad_1_1(feature_maps_s[0]).mean(dim=2)
        # x_mean_s_2 = self.revgrad_1_1(feature_maps_s[1]).mean(dim=2)
        # x_mean_s_3 = self.revgrad_1_1(feature_maps_s[2]).mean(dim=2)
        # x_mean_s_4 = self.revgrad_1_1(feature_maps_s[3]).mean(dim=2)
        # x_mean_s_5 = self.revgrad_1_1(feature_maps_s[4]).mean(dim=2)
        #
        # x_mean_1 = self.revgrad_1_1(feature_maps[0]).mean(dim=2)
        # x_mean_2 = self.revgrad_1_1(feature_maps[1]).mean(dim=2)
        # x_mean_3 = self.revgrad_1_1(feature_maps[2]).mean(dim=2)
        # x_mean_4 = self.revgrad_1_1(feature_maps[3]).mean(dim=2)
        # x_mean_5 = self.revgrad_1_1(feature_maps[4]).mean(dim=2)
        # #
        # x_std_s_1 = self.revgrad_1_1(feature_maps_s[0]).std(dim=2)
        # x_std_s_2 = self.revgrad_1_1(feature_maps_s[1]).std(dim=2)
        # x_std_s_3 = self.revgrad_1_1(feature_maps_s[2]).std(dim=2)
        # x_std_s_4 = self.revgrad_1_1(feature_maps_s[3]).std(dim=2)
        # x_std_s_5 = self.revgrad_1_1(feature_maps_s[4]).std(dim=2)
        #
        # x_std_1 = self.revgrad_1_1(feature_maps[0]).std(dim=2)
        # x_std_2 = self.revgrad_1_1(feature_maps[1]).std(dim=2)
        # x_std_3 = self.revgrad_1_1(feature_maps[2]).std(dim=2)
        # x_std_4 = self.revgrad_1_1(feature_maps[3]).std(dim=2)
        # x_std_5 = self.revgrad_1_1(feature_maps[4]).std(dim=2)

        x_mean_s = [self.revgrad_1_1(layer).mean(dim=2) for layer in feature_maps_s]
        x_mean = [self.revgrad_1_1(layer).mean(dim=2) for layer in feature_maps]

        x_std_s = [self.revgrad_1_1(layer).std(dim=2) for layer in feature_maps_s]
        x_std = [self.revgrad_1_1(layer).std(dim=2) for layer in feature_maps]

        x = torch.cat(x_mean_s + x_mean + x_std_s + x_std, dim=1)

        # x = torch.cat(
        #     [
        #         x_mean_s_1,
        #         x_mean_s_2,
        #         x_mean_s_3,
        #         x_mean_s_4,
        #         x_mean_s_5,
        #         x_mean_1,
        #         x_mean_2,
        #         x_mean_3,
        #         x_mean_4,
        #         x_mean_5,
        #         x_std_s_1,
        #         x_std_s_2,
        #         x_std_s_3,
        #         x_std_s_4,
        #         x_std_s_5,
        #         x_std_1,
        #         x_std_2,
        #         x_std_3,
        #         x_std_4,
        #         x_std_5,
        #     ],
        #     dim=1,
        # )

        x = torch.relu(self.out_s_layer_1(x))
        x = torch.sigmoid(self.out_s_layer_2(x))
        return x

    def __encoder_block(self, input):

        # first head
        x = self.stem_encoder(input)
        skip_outputs = []
        for layer in self.wavenet_encoder:
            x, skip_out = layer(x)
            skip_outputs.append(skip_out)

        x = torch.mean(torch.stack(skip_outputs), dim=0)

        return x, skip_outputs
