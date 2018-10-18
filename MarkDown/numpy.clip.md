# Numpy.clip

>numpy.clip(a, a_min, a_max, out=None)[source]
Clip (limit) the values in an array.

>Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.

a：输入矩阵
a_min：interval最小值
a_max：interval最大值
out：如果设为out=a，则表示in-place clip

# Reference
https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html