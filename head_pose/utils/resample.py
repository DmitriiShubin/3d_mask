"""
resample.py
-----------
This module provides a function for resampling ECG waveforms.
By: Dmitrii Shubin, 2020
"""

# 3rd party imports
import numpy as np
from torch import nn
import torch


class Resampling(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights_LPF_2 = torch.Tensor(
            [
                0.0003635,
                0.0003753,
                -0.0003963,
                -0.0004272,
                0.0004683,
                0.0005204,
                -0.000584,
                -0.0006597,
                0.0007481,
                0.0008498,
                -0.0009656,
                -0.001096,
                0.001242,
                0.001404,
                -0.001584,
                -0.001781,
                0.001997,
                0.002234,
                -0.002491,
                -0.002772,
                0.003076,
                0.003406,
                -0.003763,
                -0.00415,
                0.00457,
                0.005024,
                -0.005517,
                -0.006052,
                0.006635,
                0.00727,
                -0.007964,
                -0.008726,
                0.009566,
                0.0105,
                -0.01153,
                -0.0127,
                0.01402,
                0.01552,
                -0.01727,
                -0.01931,
                0.02176,
                0.02473,
                -0.02846,
                -0.03327,
                0.03976,
                0.04905,
                -0.06354,
                -0.08945,
                0.1496,
                0.4497,
                0.4497,
                0.1496,
                -0.08945,
                -0.06354,
                0.04905,
                0.03976,
                -0.03327,
                -0.02846,
                0.02473,
                0.02176,
                -0.01931,
                -0.01727,
                0.01552,
                0.01402,
                -0.0127,
                -0.01153,
                0.0105,
                0.009566,
                -0.008726,
                -0.007964,
                0.00727,
                0.006635,
                -0.006052,
                -0.005517,
                0.005024,
                0.00457,
                -0.00415,
                -0.003763,
                0.003406,
                0.003076,
                -0.002772,
                -0.002491,
                0.002234,
                0.001997,
                -0.001781,
                -0.001584,
                0.001404,
                0.001242,
                -0.001096,
                -0.0009656,
                0.0008498,
                0.0007481,
                -0.0006597,
                -0.000584,
                0.0005204,
                0.0004683,
                -0.0004272,
                -0.0003963,
                0.0003753,
            ]
        )
        self.weights_LPF_2 = self.weights_LPF_2.view(1, 1, self.weights_LPF_2.shape[0])
        self.padding_LPF_2 = int((self.weights_LPF_2.shape[2] - 1) / 2)
        self.padding_LPF_2 = torch.Tensor(np.zeros((self.padding_LPF_2)))

        self.weights_LPF_4 = torch.Tensor(
            [
                0.00023329518048871125,
                0.000524011922849899,
                0.0003961823680164898,
                -9.446639743841332e-05,
                -0.0005899206138222028,
                -0.000655532687672521,
                -0.00012915360135592652,
                0.0006594407518566544,
                0.0010445212689411249,
                0.0005454107979488138,
                -0.0006197085021846031,
                -0.001530423113025621,
                -0.001241569449267936,
                0.0003105494817994001,
                0.001994676825077854,
                0.002243236596275157,
                0.00044168657334944604,
                -0.0022327970282712654,
                -0.0034785309876846516,
                -0.001778751524827407,
                0.0019740254945049946,
                0.00475533697400548,
                0.0037618317594815646,
                -0.0009178727031122407,
                -0.00575623924409351,
                -0.006328680154576885,
                -0.001220124615009512,
                0.0060501426719602285,
                0.009263806983283038,
                0.004665563445161563,
                -0.005111159055421356,
                -0.01218363022451356,
                -0.00956207302248116,
                0.0023212046743054924,
                0.014527237044030384,
                0.015993682394147984,
                0.003099493318389358,
                -0.015516656706065328,
                -0.02410844917487526,
                -0.01239475879863364,
                0.013962445119187478,
                0.0345330247705059,
                0.02844474618749713,
                -0.007357019115044654,
                -0.05008113980913892,
                -0.061783702660840274,
                -0.014050835064989184,
                0.08941338321147038,
                0.2089283334086381,
                0.2886359250311735,
                0.2886359250311735,
                0.2089283334086381,
                0.08941338321147038,
                -0.014050835064989184,
                -0.06178370266084027,
                -0.05008113980913892,
                -0.007357019115044654,
                0.02844474618749713,
                0.0345330247705059,
                0.013962445119187478,
                -0.012394758798633639,
                -0.02410844917487526,
                -0.015516656706065324,
                0.003099493318389358,
                0.015993682394147984,
                0.014527237044030384,
                0.002321204674305491,
                -0.009562073022481157,
                -0.012183630224513558,
                -0.005111159055421356,
                0.0046655634451615614,
                0.009263806983283036,
                0.006050142671960228,
                -0.001220124615009512,
                -0.006328680154576882,
                -0.005756239244093507,
                -0.0009178727031122407,
                0.0037618317594815655,
                0.004755336974005478,
                0.001974025494504994,
                -0.001778751524827407,
                -0.003478530987684649,
                -0.002232797028271264,
                0.00044168657334944604,
                0.002243236596275157,
                0.0019946768250778514,
                0.00031054948179939993,
                -0.001241569449267936,
                -0.001530423113025621,
                -0.0006197085021846027,
                0.0005454107979488134,
                0.0010445212689411249,
                0.0006594407518566544,
                -0.00012915360135592641,
                -0.0006555326876725207,
                -0.0005899206138222028,
                -9.446639743841321e-05,
                0.0003961823680164898,
                0.0005240119228498987,
                0.00023329518048871125,
            ]
        )
        self.weights_LPF_4 = self.weights_LPF_4.view(1, 1, self.weights_LPF_4.shape[0])
        self.padding_LPF_4 = int((self.weights_LPF_4.shape[2] - 1) / 2)
        self.padding_LPF_4 = torch.Tensor(np.zeros((self.padding_LPF_4)))

    def FIR_filt(self, input, weight, padding_vector):
        input = torch.cat((input, padding_vector), 0)
        input = torch.cat((padding_vector, input), 0)
        input = input.view(1, 1, input.shape[0])
        output = torch.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        output = output.view(output.shape[2])
        return output

    def forward(self, x, order):

        if order == 2:
            x = self.FIR_filt(x, self.weights_LPF_2, self.padding_LPF_2)
        elif order == 4:
            x = self.FIR_filt(x, self.weights_LPF_4, self.padding_LPF_4)

        return x

    def upsample(self, X, order: int):

        assert order == int(2) or order == int(4), "2 and 4 order values are supported at the moment"

        if order == int(2):
            resampled = np.zeros((X.shape[0]) * 2)
            for i in range(X.shape[0]):
                resampled[i * 2] = X[i] * 2

            self.eval()
            resampled = torch.tensor(resampled, dtype=torch.float)
            resampled = resampled
            resampled = self.forward(resampled, order=2)
            resampled = resampled.numpy()

        elif order == int(4):
            resampled = np.zeros((X.shape[0]) * 4)
            for i in range(X.shape[0]):
                resampled[i * 4] = X[i] * 4

            self.eval()
            resampled = torch.tensor(resampled, dtype=torch.float)
            resampled = resampled
            resampled = self.forward(resampled, order=4)
            resampled = resampled.numpy()

        return resampled

    def downsample(self, X, order: int):

        assert order == int(2), "2nd order values are supported at the moment"

        if order == int(2):
            self.eval()
            X = torch.tensor(X, dtype=torch.float)
            X = self.forward(X, order=2)
            X = X.numpy()

            resampled = np.zeros((X.shape[0]) // 2)
            for i in range(resampled.shape[0]):
                resampled[i] = X[i * 2]

        return resampled
