import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

# load datasets
period = 1.0  # period of oscillations (seconds)
w = 50
t_min = 0
t_max = 0.1
t_step = 0.001
t = np.arange(t_min, t_max, t_step, dtype=np.float)
fai = np.pi / 4
A = np.sin(2 * np.pi * w * t)
B = np.sin(2 * np.pi * w * t + fai)

# B = np.roll(A, 2)

nsamples = A.size

plt.plot(range(nsamples), A)
plt.plot(range(nsamples), B)

# plt.show()

xcorr = correlate(A, B)
print(xcorr.shape)
xcorr_pos = np.argmax(xcorr)
print(xcorr_pos)
plt.plot(range(2*nsamples-1), xcorr)

# plt.show()

print('t:', xcorr_pos*t_step)
print('fai:', xcorr_pos*t_step * 2 * np.pi * w)


# The peak of the cross-correlation gives the shift between the two signals
# The xcorr array goes from -nsamples to nsamples
dt = np.linspace(-t[-1], t[-1], 2*nsamples-1)
recovered_time_shift = dt[xcorr.argmax()]
print(recovered_time_shift)

# force the phase shift to be in [-pi:pi]
recovered_phase_shift = 2*np.pi*(((0.5 + recovered_time_shift/period) % 1.0) - 0.5)

relative_error = (recovered_phase_shift - fai)/(2*np.pi)

print("Original phase shift: %.2f pi" % (fai/np.pi))
print("Recovered phase shift: %.2f pi" % (recovered_phase_shift/np.pi))
print("Relative error: %.4f" % (relative_error))
