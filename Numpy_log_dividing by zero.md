# “Divide by zero encountered in log” when not dividing by zero
$y = log(x)$
由于log函数是一条单调递增曲线，且在x=0处y取值为负无穷。所以在计算机中计算log时，要注意避免x=0。
计算时可以在x的基础上添加一个微笑的delta，即log(x+delta)。

# 参考
https://stackoverflow.com/questions/36229340/divide-by-zero-encountered-in-log-when-not-dividing-by-zero/36229376
