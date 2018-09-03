# Numpy.random
## numpy.random.rand(d0, d1, ..., dn)
Random values in a given shape.

Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).

Parameters:	
d0, d1, …, dn : int, optional
The dimensions of the returned array, should all be positive. If no argument is given a single Python float is returned.

Returns:	
out : ndarray, shape (d0, d1, ..., dn)
Random values.

按指定形状生成介于[0, 1)的均匀随机分布。

## numpy.random.uniform(low=0.0, high=1.0, size=None)
Draw samples from a uniform distribution.

Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high). In other words, any value within the given interval is equally likely to be drawn by uniform.

Parameters:	
low : float or array_like of floats, optional
Lower boundary of the output interval. All values generated will be greater than or equal to low. The default value is 0.

high : float or array_like of floats
Upper boundary of the output interval. All values generated will be less than high. The default value is 1.0.

size : int or tuple of ints, optional
Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if low and high are both scalars. Otherwise, np.broadcast(low, high).size samples are drawn.

Returns:	
out : ndarray or scalar
Drawn samples from the parameterized uniform distribution.

按指定形状生成介于[low, high)的均匀随机分布。

