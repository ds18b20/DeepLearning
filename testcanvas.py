#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from io import BytesIO
from PIL import Image
import base64
import cv2
import time
import functools


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))
    return wrapper


# path variables#path v
game_url = "chrome://dino"
# chrome_driver_path = r"C:\Windows\System32chromedriver"

# scripts
# create id for canvas for faster selection from DOM
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

# use id created above
# get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"


class Game(object):
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")  # mute mode
        self._driver = webdriver.Chrome( chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get('chrome://dino')
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    @timeit
    def screen_capture(self):
        image_b64 = self._driver.execute_script(getbase64Script)
        xxx = ?
        image = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        return image


def show_img(graphs=False):
    """
    show images in a new window, using Python coroutine.
    :param graphs: select window title name
    :return: None
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    game = Game()
    coroutines = show_img()
    coroutines.__next__()
    game.press_up()
    while True:
        # read from canvas
        img = game.screen_capture()
        print("raw data shape: {}".format(img.shape))

        # send new image data to coroutine
        # img = process_img(img)
        print("processed data shape: {}".format(img.shape))

        coroutines.send(img)
