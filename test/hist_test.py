import numpy as np
import matplotlib.pyplot as plt


plt.figure(0)
ax_1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax_2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax_3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax_4 = plt.subplot2grid((3, 3), (2, 0))
ax_5 = plt.subplot2grid((3, 3), (2, 1))
plt.tight_layout()

plt.figure(1)
ax_a = plt.subplot(1, 2, 1)
ax_a.set_title('a')
ax_b = plt.subplot(1, 2, 2)
ax_b.set_title('b')
plt.tight_layout()

plt.show()
