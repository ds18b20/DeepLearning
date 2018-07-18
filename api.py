#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import pickle
from PIL import Image
from io import BytesIO
import base64

# path variables#path v
game_url = "chrome://dino"
chrome_driver_path = "/anaconda3/lib/python3.6/chromedriver"
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"

# scripts
# create id for canvas for faster selection from DOM
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

# get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"


'''
* Game class: Selenium interfacing between the python and browser
* __init__():  Launch the browser window using the attributes in chrome_options
* get_crashed() : return true if the agent as crashed on an obstacles. Gets javascript variable from game decribing the state
* get_playing(): true if game in progress, false is crashed or paused
* restart() : sends a signal to browser-javascript to restart the game
* press_up(): sends a single to press up get to the browser
* get_score(): gets current game score from javascript variables.
* pause(): pause the game
* resume(): resume a paused game if not crashed
* end(): close the browser and end the game
'''


class Game(object):
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path=chrome_driver_path,chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get('chrome://dino')
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)  # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()


class DinoAgent(object):
    def __init__(self, game):  # takes game as input for taking actions
        self._game = game
        self.jump()  # to start the game, we need to jump once

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


class Game_sate:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img()  # display the processed image on screen using openCV, implemented using python coroutine
        self._display.__next__()  # initiliaze the display coroutine

    # def get_state(self, actions):
    #     actions_df.loc[len(actions_df)] = actions[1]  # storing actions in a dataframe
    #     score = self._game.get_score()
    #     reward = 0.1
    #     is_over = False  # game over
    #     if actions[1] == 1:
    #         self._agent.jump()
    #     image = grab_screen(self._game._driver)
    #     self._display.send(image)  # display the image on screen
    #     if self._agent.is_crashed():
    #         scores_df.loc[len(loss_df)] = score # log the score when game is over
    #         self._game.restart()
    #         reward = -1
    #         is_over = True
    #     return image, reward, is_over  # return the Experience tuple


def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb') as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)  # processing image as required
    return image


def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def show_img(graphs=False):
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    game = Game()
    dino = DinoAgent(game)
    # while game.get_crashed():
    count = 0
    crush_flag = game.get_crashed()
    play_flag = game.get_playing()
    print("Crush: {}".format(crush_flag))
    print("Play:  {}".format(play_flag))
    while True:
        game.press_down()
        time.sleep(0.5)
        # crush_flag = game.get_crashed()
        # play_flag = game.get_playing()
        # time.sleep(0.5)
        # game.press_up()
        # if crush_flag:
        #     count += 1
        #     time.sleep(1)
        #     game.restart()
        #     if count > 2:
        #         break
