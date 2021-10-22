import numpy as np
import numba


class Peak_finder:
    def find_peaks_abp(self, abp, threshold):

        temp = (abp - np.mean(abp)) / np.std(abp)
        peaks, _ = peakdet(temp, threshold)
        if len(peaks) > 0:
            peaks = np.array(peaks).astype(np.int64)
            peaks = peaks[:, 0]
            peaks = peaks[np.where(peaks != 0)]
            if peaks.shape[0] > 0:
                return peaks
            else:
                return None
        else:
            return None


@numba.jit(nopython=False, parallel=True)
def peakdet(v, delta: float):

    # list of peaks
    maxtab = []
    mintab = []

    x = np.arange(v.shape[0])

    assert delta > 0, 'Delta should be positive'

    mn = np.inf
    mx = -np.inf
    mnpos = np.nan
    mxpos = np.nan

    lookformax = True

    for i in range(v.shape[0]):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx - delta:
                maxtab.append([mxpos, mx])
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append([mnpos, mn])
                mx = this
                mxpos = x[i]
                lookformax = True

    return maxtab, mintab
