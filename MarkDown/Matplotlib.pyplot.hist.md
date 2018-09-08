# Matplotlib Histogram

`n, bins, patches = hist(x, bins=10, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None, **kwargs)`

range : tuple or None, optional， 表示画图的范围大小
英文详细介绍：

The lower and upper range of the bins. Lower and upper outliers are ignored. If not provided, range is (x.min(), x.max()). Range has no effect if bins is a sequence.

If bins is a sequence or range is specified, autoscaling is based on the specified bin range instead of the range of x.

Default is None

```python
import numpy as np
import matplotlib.pyplot as plt

mean = 10
sigma = 1
x = mean + sigma * np.random.randn(1000)

bins_num = 10  # bar number

n, bins, patches = plt.hist(x, bins_num, normed=True)
print(n, bins, patches)
plt.show()
```
# Reference
很好的总结，不只是Hist的介绍
https://www.cnblogs.com/yinheyi/p/6056314.html

https://www.sohu.com/a/195391558_654419
