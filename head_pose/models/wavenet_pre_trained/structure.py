import torch
import torch.nn as nn


class Wave_block(nn.Module):
    def __init__(self, out_ch, kernel_size, dilation, dropout_rate):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_ch = out_ch

        self.conv1 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            bias=False,
        )

        self.conv_res = nn.Conv1d(
            out_ch,
            out_ch,
            1,
            padding=0,
            dilation=dilation,
            bias=False,
        )

        self.conv_skip = nn.Conv1d(
            out_ch,
            out_ch,
            1,
            padding=0,
            dilation=dilation,
            bias=False,
        )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        res_x = x

        tanh = self.tanh(self.bn1(self.conv1(x)))
        sig = self.sigmoid(self.bn2(self.conv2(x)))
        res = torch.mul(tanh, sig)

        res = self.dropout(res)

        res_out = self.conv_res(res) + res_x
        skip_out = self.conv_skip(res)

        return res_out, skip_out


class Stem_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout_rate, pool_size):
        super().__init__()
        dilation = 1
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=pool_size, stride=2)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.drop(x)
        return x


class Stem_layer_upsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout_rate, scale_factor):
        super().__init__()
        dilation = 1
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=int((kernel_size + (kernel_size - 1) * (dilation - 1)) / 2),
            dilation=dilation,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class WaveNet(nn.Module):
    def __init__(
        self, hparams, encoder_block=Stem_layer, decoder_block=Stem_layer_upsample, base_block=Wave_block
    ):
        super().__init__()

        self.__base_block = base_block
        self.__encoder_block = encoder_block
        self.__decoder_block = decoder_block

        self.hparams = hparams

        self.stem_encoder = self.build_stem_encoder()
        self.wavenet_encoder = self.build_wavenet_encoder()
        self.decoder = self.__build_decoder()

        shape_out = self.hparams['n_samples']

        for i in range(len(self.hparams['layer_feature_maps'])):
            shape_out //= self.hparams['pool_size']

        for i in range(len(self.hparams['layer_feature_maps'])):
            shape_out *= self.hparams['pool_size']

        self.__out_cnn = nn.Conv1d(
            in_channels=self.hparams["layer_feature_maps"][0],
            out_channels=self.hparams["n_channels"],
            kernel_size=self.hparams["kernel_size"],
            padding=(self.hparams['n_samples'] - shape_out) // 2 + 1,
            dilation=1,
            stride=1,
            bias=False,
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.stem_encoder(x)

        skip_outputs = []
        for layer in self.wavenet_encoder:
            x, skip_out = layer(x)
            skip_outputs.append(skip_out)

        x = torch.cat(skip_outputs, dim=1)

        x = self.decoder(x)
        x = self.__out_cnn(x)

        return x

    # TODO: finish
    def get_embeddings(self, x):
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        x = torch.mean(x, dim=1)

        return x

    def build_stem_encoder(self):

        stem_layer_feature_maps = self.hparams['layer_feature_maps'].copy()
        stem_encoder = []

        # build stem layers
        for index, layer in enumerate(stem_layer_feature_maps):
            if index == 0:
                stem_encoder.append(
                    self.__encoder_block(
                        in_ch=self.hparams["n_channels"],
                        out_ch=stem_layer_feature_maps[index],
                        kernel_size=self.hparams["kernel_size"],
                        dropout_rate=self.hparams["dropout_rate"],
                        pool_size=self.hparams["pool_size"],
                    ),
                )
            else:
                stem_encoder.append(
                    self.__encoder_block(
                        in_ch=stem_layer_feature_maps[index - 1],
                        out_ch=stem_layer_feature_maps[index],
                        kernel_size=self.hparams["kernel_size"],
                        dropout_rate=self.hparams["dropout_rate"],
                        pool_size=self.hparams["pool_size"],
                    ),
                )

        return nn.Sequential(*stem_encoder)

    def build_wavenet_encoder(self):
        wavenet_dilation = self.hparams['wavenet_dilation'].copy()
        wavenet_encoder = []

        # build wavenet encoder
        for dilation in wavenet_dilation:
            wavenet_encoder.append(
                self.__base_block(
                    out_ch=self.hparams['layer_feature_maps'][-1],
                    kernel_size=self.hparams["kernel_size"],
                    dilation=dilation,
                    dropout_rate=self.hparams["dropout_rate"],
                ),
            )

        return mySequential(*wavenet_encoder)

    def __build_decoder(self):

        stem_layer_feature_maps = self.hparams['layer_feature_maps'].copy()
        stem_layer_feature_maps.reverse()

        # calclulate number of input layers
        input_featurs = len(self.hparams['wavenet_dilation']) * stem_layer_feature_maps[0]

        decoder = []

        # build stem layers
        for index, layer in enumerate(stem_layer_feature_maps):
            if index == 0:
                decoder.append(
                    self.__decoder_block(
                        in_ch=input_featurs,
                        out_ch=stem_layer_feature_maps[index],
                        kernel_size=self.hparams["kernel_size"],
                        dropout_rate=self.hparams["dropout_rate"],
                        scale_factor=self.hparams["pool_size"],
                    ),
                )
            else:
                decoder.append(
                    self.__decoder_block(
                        in_ch=stem_layer_feature_maps[index - 1],
                        out_ch=stem_layer_feature_maps[index],
                        kernel_size=self.hparams["kernel_size"],
                        dropout_rate=self.hparams["dropout_rate"],
                        scale_factor=self.hparams["pool_size"],
                    ),
                )

        return nn.Sequential(*decoder)
