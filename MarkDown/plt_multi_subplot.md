# plt subplots

记录一下实现在一个figure中显示多张图表的方式。

```python
import matplotlib.pyplot as plt

# 方式一
plt.figure(0)
ax_1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax_2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax_3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax_4 = plt.subplot2grid((3, 3), (2, 0))
ax_5 = plt.subplot2grid((3, 3), (2, 1))
plt.tight_layout()

# 方式二
# 使用set_title方法显示title
plt.figure(1)
ax_a = plt.subplot(1, 2, 1)
ax_a.set_title('a')
ax_b = plt.subplot(1, 2, 2)
ax_b.set_title('b')
plt.tight_layout()

plt.show()
```

# Reference
https://stackoverflow.com/questions/37424530/how-to-make-more-than-10-subplots-in-a-figure/37444059
