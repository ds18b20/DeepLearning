import numpy as np
import scipy
from scipy.signal import correlate
import matplotlib.pyplot as plt

# load datasets
A = [-80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -79.58, -79.55, -79.08, -78.95, -78.77, -78.45,-77.75, -77.18, -77.08, -77.18, -77.16, -76.6, -76.34, -76.35]
B = [-80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -78.74, -78.65, -78.08, -77.75, -77.31, -76.55, -75.55, -75.18, -75.34, -75.32, -75.43, -74.94, -74.7, -74.68]

A = np.array(A)
B = np.array(B)

nsamples = A.size

plt.plot(range(nsamples), A)
plt.plot(range(nsamples), B)

# plt.show()

# regularize datasets by subtracting mean and dividing by s.d.
A -= A.mean(); A /= A.std()
B -= B.mean(); B /= B.std()

plt.plot(range(nsamples), A)
plt.plot(range(nsamples), B)

# plt.show()

# Put in an artificial time shift between the two datasets
time_shift = 20
A = np.roll(A, time_shift)
print(A.shape)

# Find cross-correlation
xcorr = correlate(A, B)
print(xcorr.shape)
print(xcorr)
print(np.argmax(xcorr))

# delta time array to match xcorr
dt = np.arange(1-nsamples, nsamples)
print(dt)
print(dt[19])


