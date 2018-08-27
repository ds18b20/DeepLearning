#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from pynput.keyboard import Key, Controller

keyboard = Controller()

chrome_driver_path = "/anaconda3/lib/python3.6/chromedriver"


if __name__ == "__main__":
    driver = webdriver.Chrome(executable_path=chrome_driver_path)
    # url = "https://novel12.com/"
    url = "chrome://dino"
    driver.get(url)
    keyboard.press(Key.up)
    keyboard.release(Key.up)
    time.sleep(0.5)
    count = 0
    while count < 10:
        # game.press_down()

        keyboard.press(Key.up)
        keyboard.release(Key.up)
        time.sleep(0.5)

        # keyboard.press(Key.down)
        # keyboard.release(Key.down)
        # time.sleep(0.5)
        count += 1
