# http://www.emilygraceripka.com/blog/16
# https://python4esac.github.io/fitting/examples1d.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def _2dgauss(x, H, A, x0, sigma, A2, x02, sigma2):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + A2 * np.exp(-(x - x02) ** 2 / (2 * sigma2 ** 2))


df = pd.read_csv('HD163296-co-spectral-region-1.txt', sep="\t", header=None)
# df.columns = ["x", "y"]
xarry = np.array(df[0])
yarry = np.array(df[1])

popt, pcov = curve_fit(_2dgauss, xarry, yarry, p0=[0,0.06,230.53457,0.0004,0.07,230.532592047,0.0004])
perr = np.sqrt(np.diag(pcov))

print('print [y, amp1, cen1, fwhm1, amp2, cen2, fwhm2]',popt[0],popt[1],popt[2],popt[3]*2.355,popt[4],popt[5],popt[6]*2.355)
print('print [y_err, amp_err1, cen_err1, fwhm_err1, amp_err2, cen_err2, fwhm_err2]', perr[0], perr[1], perr[2], perr[3]*2.355,perr[4], perr[5], perr[6]*2.355)

ym = _2dgauss(xarry, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])

plt.plot(xarry, ym, c='r', label='Best fit')
plt.plot(xarry, yarry, 'k-', label='data')
plt.show()