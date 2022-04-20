####################################################################################
# Simple script that goes through a corpus of unicode-encoded Tibetan texts,
# calculates the word frequencies and saves them.
####################################################################################
import json
import multiprocessing
import re
from glob import glob
from pathlib import Path

# 3rd party libraries - these need to be installed with pip install ...
import numpy as np
from botok import WordTokenizer  # botok Tibetan tokenizer
from pqdm.processes import pqdm

# Corpus folder with Tibetan texts. This folder should contain .txt files with Tibetan unicode text.
# Various Tibetan corpora can be downloaded, for example, from the following sources:
# BDRC corpus: https://zenodo.org/record/821218
# Esukhia Derge Kangyur: https://github.com/Esukhia/derge-kangyur/tree/master/text
# Esukhia Derge Tengyur: https://github.com/Esukhia/derge-tengyur/tree/master/text
corpus_folder = './corpora'

# Go through the corpus, tokenize every file and count the frequency for each token
# this is done in a parallelized way to speed up processing.
file_names = glob(f"{corpus_folder}/**/*.txt", recursive=True)
np.random.shuffle(file_names)  # randomize file order to improve the progress indication for files with varying sizes
word_frequencies = multiprocessing.Manager().dict()
file_count = multiprocessing.Manager().Value('i', 0)
update_lock = multiprocessing.Lock()
wt = WordTokenizer()


# write JSON with word frequencies
def save_file():
    word_freqs = word_frequencies.copy()  # covert multiprocessing dict to regular dict
    word_freqs = dict(
        sorted(word_freqs.items(), key=lambda keyVal: keyVal[1], reverse=True))  # sort by frequency
    with open(f'word_frequencies.json', 'w') as fp:
        json.dump(word_freqs, fp, ensure_ascii=False, indent=1)


def clean_string(str):
    str = str.replace('\n', '')
    str = str.replace(' ', '')
    str = re.sub(r'\[[^\]]*\]', '', str)
    str = re.sub(r'\([^\)]*\)', '', str)
    str = re.sub(r'\{[^\}]*\}', '', str)
    return str

def is_valid_token(str):
    return not re.match('.*[a-zA-Z0-9].*', str)

def process_file(file_name):
    # print(file_name)
    text = Path(file_name).read_text()
    text = clean_string(text)
    tokens = wt.tokenize(text)

    # Count how often each token occurs in the current file. This will reduce the amount of updates we have to do just
    # below while when holding the lock on the global frequency list.
    file_frequencies = {}
    for token in tokens:
        token_txt = token.text
        if is_valid_token(token_txt):
            if token_txt in file_frequencies:
                file_frequencies[token_txt] += 1
            else:
                file_frequencies[token_txt] = 1

    # update the global token frequencies:
    with update_lock:
        file_count.value += 1
        for token_freq in file_frequencies.items():
            token_txt = token_freq[0]
            if token_txt in word_frequencies:
                word_frequencies[token_txt] += token_freq[1]
            else:
                word_frequencies[token_txt] = token_freq[1]
        if file_count.value % 100 == 0:
            print(f'File {file_count.value} - {len(word_frequencies)} tokens total')
            save_file()


print(f"Will process {len(file_names)} files:")
pqdm(file_names, process_file, multiprocessing.cpu_count())
save_file()
