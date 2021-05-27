import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    # print("print, p0 mean:",mean)
    # print("print, p0 sigma:",sigma)
    popt, pcov = curve_fit(gauss, x, y)#, p0=[min(y), max(y), mean, sigma])
    perr = np.sqrt(np.diag(pcov))
    print('print [y, amp, cen, fwhm]',popt[0],popt[1],popt[2],popt[3]*2.355)
    print('print [y_err, amp_err, cen_err, fwhm_err]', perr[0], perr[1], perr[2], perr[3]*2.355)
    print('======================')
    return popt


df = pd.read_csv('ngc3079-co-spectral-region-2.txt', sep="\t", header=None)
xarry = np.array(df[0])
yarry = np.array(df[1])
H, A, x0, sigma = gauss_fit(xarry, yarry)


plt.plot(xarry, yarry, 'k-', label='data')
plt.plot(xarry, gauss(xarry, *gauss_fit(xarry, yarry)), '--r', label='fit')

plt.legend()
plt.title('Gaussian fit')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Value (Jy/beam)')
plt.show()

exit