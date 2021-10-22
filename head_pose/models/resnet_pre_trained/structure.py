import torch
import torch.nn as nn


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


class CRD_layer_upsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, drop_rate, scale_factor, dilation):
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
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
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

        x = self.upsample(x)

        identity = x

        x = self.relu(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.relu(x)

        x += identity

        x = self.conv3(x)
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, hparams, encoder_block=CBRD_layer, decoder_block=CRD_layer_upsample):
        super().__init__()

        dilation = 1

        self.__encoder_block = encoder_block
        self.__decoder_block = decoder_block
        self.hparams = hparams

        self.encoder = self.__build_encoder()
        self.decoder = self.__build_decoder()

        shape_out = self.hparams['n_samples']

        for i in range(len(self.hparams['layer_feature_maps'])):
            shape_out //= self.hparams['pool_size']

        for i in range(len(self.hparams['layer_feature_maps'])):
            shape_out *= self.hparams['pool_size']

        self.__out_cnn = nn.Conv1d(
            in_channels=self.hparams["n_channels"],
            out_channels=self.hparams["n_channels"],
            kernel_size=self.hparams["kernel_size"],
            padding=(self.hparams['n_samples'] - shape_out) // 2 + 1,
            dilation=dilation,
            stride=1,
            bias=False,
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.encoder(x)

        x = self.decoder(x)
        x = self.__out_cnn(x)

        return x

    def get_embeddings(self, x):
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        x = torch.mean(x, dim=1)

        return x

    def __build_encoder(self):

        layer_feature_maps = self.hparams['layer_feature_maps'].copy()
        encoder = []

        for index, layer in enumerate(layer_feature_maps):
            if index == 0:
                encoder.append(
                    # f"encoder_layer_{index}",
                    self.__encoder_block(
                        in_ch=self.hparams["n_channels"],
                        out_ch=layer_feature_maps[index],
                        kernel_size=self.hparams["kernel_size"],
                        drop_rate=self.hparams["dropout_rate"],
                        pool_size=self.hparams["pool_size"],
                        dilation=self.hparams["dilation"],
                    ),
                )
            else:
                encoder.append(
                    # f"encoder_layer_{index}",
                    self.__encoder_block(
                        in_ch=layer_feature_maps[index - 1],
                        out_ch=layer_feature_maps[index],
                        kernel_size=self.hparams["kernel_size"],
                        drop_rate=self.hparams["dropout_rate"],
                        pool_size=self.hparams["pool_size"],
                        dilation=self.hparams["dilation"],
                    ),
                )

        return nn.Sequential(*encoder)

    def __build_decoder(self):

        layer_feature_maps = self.hparams['layer_feature_maps'].copy()
        layer_feature_maps.reverse()

        decoder = []

        for index, layer in enumerate(layer_feature_maps):
            if index == len(layer_feature_maps) - 1:
                decoder.append(
                    # f"decoder_layer_{index}",
                    self.__decoder_block(
                        in_ch=layer_feature_maps[index],
                        out_ch=self.hparams['n_channels'],
                        kernel_size=self.hparams["kernel_size"],
                        drop_rate=self.hparams["dropout_rate"],
                        scale_factor=self.hparams["pool_size"],
                        dilation=self.hparams["dilation"],
                    )
                )

            else:
                decoder.append(
                    # f"decoder_layer_{index}",
                    self.__decoder_block(
                        in_ch=layer_feature_maps[index],
                        out_ch=layer_feature_maps[index + 1],
                        kernel_size=self.hparams["kernel_size"],
                        drop_rate=self.hparams["dropout_rate"],
                        scale_factor=self.hparams["pool_size"],
                        dilation=self.hparams["dilation"],
                    ),
                )

        return nn.Sequential(*decoder)
