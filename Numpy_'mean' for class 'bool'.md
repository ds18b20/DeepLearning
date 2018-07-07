# Unresolved attribute reference 'mean' for class 'bool'
    (y_hat.argmax(axis=1) == y).mean()
    # or
    np.mean(y_hat.argmax(axis=1) == y)
上述代码会提示bool对象不包含mean方法。
> Unresolved attribute reference 'mean' for class 'bool'

原因是PyCharm不能很好的识别numpy.ndarray对象，也就把上述==计算的结果存储为bool对象，进而提示bool不包含mean方法。
解决方法，可以在表达式后面加上# type: np.ndarray来提示IDE表达式结果的类型。

    tmp = y_hat.argmax(axis=1) == y  # type: np.ndarray
    np.mean(tmp)

----------
# 参考
Expected type 'Union[ndarray, Iterable]' warning in Python instruction
https://stackoverflow.com/questions/30599676/expected-type-unionndarray-iterable-warning-in-python-instruction
