####################################################################################
# Simple script that gives a summary about the overall size of a Tibetn corpus
# and the average number of bytes per token.
####################################################################################
import json
import os
from glob import glob

import numpy as np

# Corpus folder with Tibetan texts.
corpus_folder = './corpora'
# file that contains the frequency of all encountered tokens in the corpus
word_list_file = 'word_frequencies.json'


# Draw a text mode histogram. Derived from https://gist.github.com/tammoippen/4474e838e969bf177155231ebba52386
def crappyhist(data, title, max_value, bins=100, width=140):
    print()
    print('=' * len(title))
    print(title)
    print('=' * len(title))
    print()

    print("Average: {:.2f}".format(np.average(data)))
    print("Median: {:.2f}".format(np.median(data)))

    h, b = np.histogram(data, bins, range=(0, max_value))
    rest = sum(1 for dat in data if dat > max_value)
    max_count = max(np.amax(h), rest)

    for i in range(0, bins):
        print('{:7.0f}  | {:{width}s} {}'.format(
            b[i],
            '█' * int(width * h[i] / max_count),
            h[i],
            width=width))

    print('>{:7.0f} | {:{width}s} {}'.format(
        max_value,
        '█' * int(width * rest / max_count),
        rest,
        width=width))


# Check how big the corpus files are
file_names = glob(f"{corpus_folder}/**/*.txt", recursive=True)
file_sizes = [os.path.getsize(file_name) for file_name in file_names]

crappyhist(file_sizes, "File Size Distribution [bytes]:", max_value=500000)

# Get the tokens frequencies
with open(word_list_file) as fp:
    vocab = json.loads(fp.read())

token_frequency = crappyhist(list(vocab.values()), "Token Frequence:", max_value=1000, bins=100)

# Get total number of tokens in corpus and avg. amount of bytes in corpus
print("Total corpus size: {:.2f}M".format(sum(file_sizes) / (1024 * 1024)))
print("File count: {:.0f}M".format(len(file_sizes)))
print("Avg File Size: {:.0f}K".format(np.average(file_sizes) // 1024))
print("Median File Size: {:.0f}K".format(np.median(file_sizes) // 1024))
print()
print(f"Vocab size: {len(vocab)}")
print(f"Total number of tokens in corpus: {sum(vocab.values())}")

token_lengths = [len(tok) for tok in vocab.keys()]
print("Avg char size of Token:".format(np.average(token_lengths)))
print("Median char size of Token: {:.0f}".format(np.median(token_lengths)))
print("Avg bytes per token: {:.2f}".format(sum(file_sizes) / sum(vocab.values())))
