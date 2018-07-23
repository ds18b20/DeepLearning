# Python实现协程

```python:coroutine

def printer():
    counter = 0
    while True:
        string = (yield)
        print("[{0}] {1}".format(counter, string))
        counter += 1
if __name__ == "__main__":
    p = printer()
    # p.__next__()
    next(p)

    p.send("Hello")
    p.send("World")
    p.send("!")
```
上述代码使用生成器（yield）实现了协程（coroutine）。
主程序通过send函数唤起子程序并且传入数据，子程序的逻辑为保持死循环接受string并打印。
子程序接收到数据并打印之后，用yield把自己挂起，然后返回主程序。

还需要继续学习。。。

# 参考
http://www.hsfzxjy.site/python-generator-coroutine/