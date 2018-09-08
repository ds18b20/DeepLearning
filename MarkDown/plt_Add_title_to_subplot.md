# Add title to subplot

ax.set_title() should set the titles for separate subplots:
```python
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]

    fig = plt.figure()
    fig.suptitle("Title for whole figure", fontsize=16)
    ax = plt.subplot("211")
    ax.set_title("Title for first plot")
    ax.plot(data)

    ax = plt.subplot("212")
    ax.set_title("Title for second plot")
    ax.plot(data)

    plt.show()
```

# Reference
https://stackoverflow.com/questions/25239933/how-to-add-title-to-subplots-in-matplotlib
