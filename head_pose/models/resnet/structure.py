import torch
import torch.nn as nn
from torch.autograd import Function
from models.resnet_pre_trained.model import Model as Resnet_pre_trained


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


class ResNet(nn.Module):
    def __init__(
        self,
        hparams,
    ):
        super().__init__()

        # TODO: add freezeing option

        self.hparams = hparams

        # build encoder from pre-trained bottleneck model
        self.encoder = self.__build_encoder()

        # probing to calculate the number of layers
        dummy = torch.Tensor([0] * hparams['n_samples']).view(1, 1, -1)
        result = self.encoder(dummy)
        out_shape = result.shape[-2]

        self.out_layer = nn.Linear(out_shape, hparams['n_classes'])

        self.revgrad = RevGrad()
        self.out_s_layer = nn.Linear(out_shape, 1)

        # freeze encoder
        if hparams['freeze_layers']:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x, x_s=None, train=False):

        x = x.permute(0, 2, 1)

        if train:

            x_s = x_s.permute(0, 2, 1)
            out = self.main_head(x)
            adv_out = self.adv_head(x_s)

            return out, adv_out

        else:
            # TODO: try both sequential and parallel options
            out = self.main_head(x)
            return out

    def main_head(self, x):
        x = self.encoder(x)
        x_mean = torch.mean(x, dim=2)
        x_max = torch.amax(x, dim=2)

        x = x_mean + x_max
        x = torch.sigmoid(self.out_layer(x))
        return x

    def adv_head(self, x):
        x = self.encoder(x)
        x = self.revgrad(x)
        x_mean = torch.mean(x, dim=2)
        x_max = torch.amax(x, dim=2)
        x = x_mean + x_max
        x = torch.sigmoid(self.out_s_layer(x))
        return x

    def __build_encoder(self):

        AE = Resnet_pre_trained.restore(self.hparams['pre_trained_model'])

        return AE.model.encoder
