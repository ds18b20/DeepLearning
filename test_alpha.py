import os
import urllib.request
import io
"""
file_save_path = "datasets\\text\\US_Cities.txt"
us_cities_url = "https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt"
if not os.path.isfile(file_save_path):
    urllib.request.urlretrieve(url=us_cities_url, filename=file_save_path)
"""
file_save_path = r"datasets\text\US_Cities.txt"
shakespeare_input_url = "https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/shakespeare_input.txt"
if not os.path.isfile(file_save_path):
    urllib.request.urlretrieve(url=shakespeare_input_url, filename=file_save_path)

with io.open(file_save_path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('chars[0:20]:', chars[0:20])
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
