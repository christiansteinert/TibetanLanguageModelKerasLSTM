####################################################################################
# Train an LSTM language model for Tibetan.
# This code can run by itself or be copied into a Jupyter Notebook.
####################################################################################

colab_resource_monitor = False  # should an external resource monitor be used on Google Colab?
lr_find = False  # try to find a good learning rate instead of actually training the model?
only_predict = False  # Only predict instead of continuing to train?
report_file_load_times = False  # report file sizes and file load times?

import builtins
import os
import pip

folder_prefix = ''
temp_folder = '/tmp/'
tokenizer_model = 'tokenizers/bo_classical-bpe.model'
corpus_folder = 'corpora/BDRC/'

# initialization for working inside jupyter
jupyter_active = getattr(builtins, "__IPYTHON__", False)
if jupyter_active:  # are we running inside Jupyter?
    if 'google.colab' in str(get_ipython()):  # are we running on Google Colab?
        from google.colab import drive

        drive.mount('/content/drive')
        folder_prefix = '/content/drive/MyDrive/Colab Notebooks/'
        temp_folder = '/var/tmp/'
        corpus_zipped = folder_prefix + 'data/corpora/'
        corpus_folder = temp_folder + 'corpora/'
        tokenizer_model = folder_prefix + 'data/models/tokenizers/bo_classical-bpe.model'

        # unpack all data from Google Drive into a local temp folder
        # This makes file access much faster
        if not os.path.exists(corpus_folder):
            current_dir = os.getcwd()
            os.makedirs(corpus_folder, exist_ok=True)
            os.chdir(corpus_folder)
            os.system(f"unzip '{corpus_zipped}*.zip'")
            os.chdir(current_dir)

        # monitor colab CPU and GPU resource usage:
        if colab_resource_monitor:
            pip.main(['install', 'psutil==5.9.0'])
            from urllib.request import urlopen

            exec(urlopen("http://colab-monitor.smankusors.com/track.py").read())
            _colabMonitor = ColabMonitor().start()

            # Don't pre-allocate memory; allocate as-needed. This helps to better see how much GPU memory we are *actually* using
            # import tensorflow as tf
            # for dev in tf.config.experimental.list_physical_devices('GPU'):
            #    tf.config.experimental.set_memory_growth(dev, True)

        pip.main(['install', 'keras_lr_finder'])
        pip.main(['install', 'botok'])

import itertools
import json
import re
import time
from glob import glob
from typing import List, Callable

import numpy as np
import tensorflow.keras.utils as ku
from botok import WordTokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.optimizer_v2.nadam import Nadam
from keras.preprocessing.sequence import pad_sequences
from prettytable import PrettyTable
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from threading import Lock
from keras_lr_finder import LRFinder

# define important hyperparameters
hyperparameters = {
    'min_word_frequency': 10,
    # How often must a word at least have occurred in the corpus to be included into the vocabulary rather than treated as <UNK>
    'tokens_per_batch': 3000,  # how many tokens per batch?
    'learning_rate': 0.001,  # learning rate to be used
    'epochs': 20,  # how many epochs of training should be done?
    'dropout': 0.1,  # how much dropout should be added?
    'max_sequence_len': 100,  # how long do we want to make a sequence at most?
    'valid_pct': 0.005,  # fraction of the data to use as validation set
    'files_percentage': 0.05,
    # how much of the corpus should be used during each epoch to speed up small experiments?
    'file_name': folder_prefix + 'lang_model',  # name for saving / loading the language model
    'max_file_chunk_size': 20 * 1024
    # max allowed chunk size (in chars) to read at once. Large amuonts will increase memory use and will freeze Google Colab
}


class TibetanTokenizer:
    def __init__(self, word_list_file='word_frequencies.json', vocab_size=30000, min_word_frequency=-1):
        self.wt = WordTokenizer()

        with open(word_list_file) as fp:
            vocab = json.loads(fp.read())

        self._vocab = vocab

        vocab_values = list(self._vocab.values())
        self._tokens_in_corpus = np.sum(vocab_values)

        if min_word_frequency > 0:
            vocab = dict(filter(lambda item: item[1] >= min_word_frequency, self._vocab.items()))
            self._vocab_size = len(vocab)
            self._unk_probability = np.sum(
                [freq for freq in vocab_values if freq < min_word_frequency]) / self._tokens_in_corpus
        else:
            self._vocab_size = min(len(vocab), vocab_size - 2)
            if len(vocab) > self._vocab_size:
                vocab = dict(itertools.islice(vocab.items(), 0, self._vocab_size))
            self._unk_probability = np.sum(vocab_values[vocab_size:]) / self._tokens_in_corpus

        # add tokens for unknown token and beginning of sequence
        vocab['<UNK>'] = -1
        vocab['<BOS>'] = -1
        self._vocab_size += 2

        # create lookup dictionaries for translating tokens to numbers and back
        self._idx_to_token = list(vocab.keys())
        self._token_to_idx = {}
        for i in range(0, self._vocab_size):
            self._token_to_idx[self._idx_to_token[i]] = i

    def general_token_probability(self, token_num: int):
        token_str = self.de_numericalize([token_num])
        if token_str[0] == '<UNK>':
            return self._unk_probability
        else:
            return self._vocab[token_str[0]] / self._tokens_in_corpus

    def tokenize(self, text: str, add_bos=True) -> List[str]:
        tokens = self.wt.tokenize(text, split_affixes=False)
        result = [token.text for token in tokens]
        result = list(filter(lambda token: self._is_valid_token(token), result))

        if add_bos:
            result.insert(0, '<BOS>')

        return result

    def tokenize_num(self, text: str, add_bos=True) -> List[int]:
        return self.numericalize(self.tokenize(text, add_bos=add_bos))

    # convert a list of tokens into number, based in the vocabulary of the tokenizer
    def numericalize(self, tokens: List[str]) -> List[int]:
        unk = self._token_to_idx.get('<UNK>')
        return [self._token_to_idx.get(t, unk) for t in tokens]

    def get_unk_idx(self) -> int:
        return self._token_to_idx.get('<UNK>')

    # convert a list of token-numbers back into tokens
    def de_numericalize(self, token_nums: List[int]) -> List[str]:
        return [self._idx_to_token[n] for n in token_nums]

    def vocab_size(self) -> int:
        return self._vocab_size

    def _is_valid_token(self, txt: str):
        return not re.match('.*[a-zA-Z0-9].*', txt)


# This class allows to read through a list of files and provides the tokens for them.
# Even though this is implemented as a "Sequence", it does not allow random access and instead only returns chunks
# from files in a random order or one after another. So no matter which index is requested, this "Sequence"
# implementation will always supply the next batch in line in either random or sequential order.
# If used within a multiprocessing environment, each process will use a different random seed to supply the
# content of different files from each process.
class TextFilesSequence(Sequence):
    lock = Lock()

    def __init__(self, input_file_names: str, file_importer: Callable, average_bytes_per_token: float = 13.0,
                 max_sequence_len: int = 100, vocab_size: int = 30000, tokens_per_batch: int = 10000,
                 random_order=True, files_percentage: float = 0.1, report_file_load_times=False):
        self._file_number = -1
        self._iterator_position = -1
        self._file_pos = 0

        self._x = np.array([])
        self._y = np.array([])

        self._file_names = input_file_names
        self._file_importer = file_importer
        self._tokens_per_batch = tokens_per_batch
        self._files_pct = files_percentage
        self._vocab_size = vocab_size
        self._max_sequence_len = max_sequence_len
        self._average_bytes_per_token = average_bytes_per_token
        self._total_bytes = sum(self._get_file_size(file_name) for file_name in input_file_names)
        self._process_randseed_initialized = -1
        self._report_file_load_times = report_file_load_times
        self._random_order = random_order

        self.on_epoch_end()

    def __len__(self):
        # estimate the overall length of the sequence
        tokens_estimated = self._total_bytes // self._average_bytes_per_token
        return int(tokens_estimated * self._files_pct // self._tokens_per_batch)

    def __getitem__(self, idx):
        # We use multiprocessing. Each TextFilesSequence should have its own
        # random state. Therefore we should set a new random seed for each process
        if self._process_randseed_initialized != os.getpid():
            np.random.seed((os.getpid() * time.time_ns()) % 123456789)
            self._process_randseed_initialized = os.getpid()

        self._iterator_position += 1
        # print(f"get:{self._iterator_position}")

        # read as many files as needed to get the required tokens of a batch
        total_bytes_read = 0
        start_time = time.time()

        while len(self._x) <= self._tokens_per_batch:
            if self._file_pos == 0:
                if self._random_order:
                    self._file_number = np.random.random_integers(0, len(self._file_names) - 1)
                else:
                    self._file_number += 1
                    if self._file_number >= len(self._file_names):
                        self._file_number = 0

            if self._report_file_load_times:
                file_size = os.path.getsize(self._file_names[self._file_number])

                if total_bytes_read == 0:
                    indicator = '>'
                else:
                    indicator = ' '
                total_bytes_read += file_size

                print(
                    f" PID: {indicator}{os.getpid()}, getitem {idx}, file:{self._file_number} {self._file_pos}, itpos: {self._iterator_position}, size: {file_size // 1024}K, name: {self._file_names[self._file_number]}")

            x, y, self._file_pos = self._file_importer(self._file_names[self._file_number], self._file_pos)

            if len(self._x) == 0:
                self._x = x
                self._y = y
            else:
                self._x = np.append(self._x, x, axis=0)
                self._y = np.append(self._y, y, axis=0)

        # return the expected amount of tokens
        result = self._x[:self._tokens_per_batch, :], self._y[:self._tokens_per_batch, :]

        self._x = self._x[self._tokens_per_batch:, :]
        self._y = self._y[self._tokens_per_batch:, :]

        if self._report_file_load_times and total_bytes_read > 0:
            print(
                f" PID: <{os.getpid()}, getitem {idx} done, {total_bytes_read // 1024}K, {int(time.time() - start_time)} seconds")

        return result

    def on_epoch_end(self):
        # reset the current file position, iterator position and token buffer
        self._file_number = -1
        self._iterator_position = -1
        self._file_pos = 0

        if not self._random_order:
            self._x = np.array([])
            self._y = np.array([])

    def _get_file_size(self, file_name):
        return os.path.getsize(file_name)


class LstmLanguageModelTrainer:
    def __init__(self, hyperparameters, tokenizer):
        self._model = None
        self._hyperparameters = hyperparameters
        self._tokenizer = tokenizer

    def _clean_string(self, txt: str):
        return re.sub(r'\n| |\[[^]]*\]|\([^)]*\)|\{[^}]*\}', '', txt)

    def _import_file(self, file_name: str, pos: int):
        # tokenization
        vocab_size = self._tokenizer.vocab_size()
        max_sequence_len = self._hyperparameters['max_sequence_len']

        # create input sequences using list of tokens
        input_sequences = []
        with open(file_name, 'r') as f:
            if pos > 0:
                f.seek(pos)

            max_chunk_size = self._hyperparameters['max_file_chunk_size']
            text = ''
            last_char = '_'
            chunk_len = 0

            # Read the file until we have reached the first shad after the maximum allowed chunk size
            # or until we have encountered the end of the file
            while chunk_len < max_chunk_size or last_char != '།':
                last_char = f.read(1)
                if last_char != '':
                    text += last_char
                    chunk_len += 1
                else:
                    break

            if last_char == '':
                pos = 0  # end of file was reached
            else:
                pos = f.tell()  # we are still within  the file and need to continue reading next time

            text = self._clean_string(text)
            token_list = self._tokenizer.tokenize_num(text)

            for i in range(1, len(token_list)):
                seq_start = max(0, i + 1 - max_sequence_len)
                n_gram_sequence = token_list[seq_start:i + 1]
                input_sequences.append(n_gram_sequence)

        # pad sequence
        input_sequences = np.array(
            pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre', truncating='pre'))

        # create predictors and label
        predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
        label = ku.to_categorical(label, num_classes=vocab_size, dtype=np.int8)

        return predictors, label, pos

    def create_model(self):
        ### Define model ###
        model = Sequential()
        vocab_size = self._tokenizer.vocab_size()
        model.add(Embedding(vocab_size, 50, input_length=self._hyperparameters['max_sequence_len'] - 1))

        # Option 1: two LSTM layers with dropout
        # model.add(LSTM(150, return_sequences=True))
        # model.add(Dropout(self._hyperparameters['dropout']))
        # model.add(LSTM(150))

        # Option 2: a single, larger LSTM layer
        model.add(LSTM(300))

        model.add(Dense(vocab_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Nadam(learning_rate=self._hyperparameters['learning_rate']),
                      metrics=['accuracy'])
        self._model = model

    def lr_find(self, training):
        # try out different learning rates to see which one might be a good starting point
        lr_finder = LRFinder(self._model)

        # The learning rate finder cannot handle Sequence objects.
        # Therefore, we need to prefetch multiple batches and combine them
        batch_count = min(30, len(training))
        x = np.concatenate([training[i][0] for i in range(0, batch_count)])
        y = np.concatenate([training[i][1] for i in range(0, batch_count)])

        lr_finder.find(x, y, start_lr=0.0001, end_lr=1, epochs=5,
                       batch_size=1000)
        lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)

        table = PrettyTable()
        table.field_names = ["Learning Rate", "Loss"]
        for i in range(0, len(lr_finder.lrs)):
            table.add_row([lr_finder.lrs[i], lr_finder.losses[i]])
        print(table)

    def train_model(self, training: object, validation: object):
        # handler for saving the model during training
        fname = f'{self._hyperparameters["file_name"]}_train'
        autosave_handler = ModelCheckpoint(fname,
                                           verbose=1,
                                           save_best_only=True,
                                           monitor='val_loss',
                                           mode='min')
        self._save_hyperparams(fname)

        # train the model
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self._model.fit(x=training,
                        validation_data=validation,
                        epochs=self._hyperparameters['epochs'],
                        verbose=1,
                        callbacks=[earlystop, autosave_handler],
                        workers=6,
                        use_multiprocessing=True)

    def get_model(self):
        return self._model

    def load_model(self, fname: str):
        try:
            self._model = load_model(fname)
            self._load_hyperparams(fname)
            return True
        except Exception as e:
            print('Error while trying to load model:', e)
            return False

    def _load_hyperparams(self, name: str):
        with open(f'{name}.json') as fp:
            self._hyperparameters = json.loads(fp.read())

    def save_model(self, name: str):
        self._model.save(name)

    def _save_hyperparams(self, name: str):
        # save hyperparameters
        with open(f'{name}.json', 'w') as fp:
            json.dump(self._hyperparameters, fp)

    def get_data(self, text_file_pattern: str, rand_seed=89674894, report_file_load_times=False):
        # read file names
        print('scanning files')
        text_files = glob(text_file_pattern, recursive=True)

        # bring the files into a randomized order but use a random seed to have a stable order if the program is
        # Re-run and continues training an existing order.
        # A reproducible "random" order of the file names is important because the split into training and validation
        # sets depends on this.
        text_files.sort()
        rand_state = np.random.get_state()
        np.random.seed(rand_seed)
        np.random.shuffle(text_files)
        np.random.set_state(rand_state)  # restore previous randomizer state

        # split files into training and validation set
        validation_split = int(self._hyperparameters['valid_pct'] * len(text_files))
        validation = text_files[:validation_split]
        training = text_files[validation_split:]

        print(f'found {len(text_files)} files: {len(training)} for training / {len(validation)} for validation')

        tokens_per_batch = self._hyperparameters['tokens_per_batch']
        pct = self._hyperparameters['files_percentage']
        vs = self._tokenizer.vocab_size()

        train_seq = TextFilesSequence(training, self._import_file, tokens_per_batch=tokens_per_batch,
                                      files_percentage=pct, vocab_size=vs, random_order=True,
                                      report_file_load_times=report_file_load_times)
        val_seq = TextFilesSequence(validation, self._import_file, tokens_per_batch=tokens_per_batch,
                                    random_order=False, files_percentage=1.0, vocab_size=vs,
                                    report_file_load_times=report_file_load_times)
        return train_seq, val_seq

    def generate_text(self, seed_text: str, words_to_generate: int):
        unk_idx = self._tokenizer.get_unk_idx()
        for _ in range(words_to_generate):
            token_list = self._tokenizer.tokenize_num(seed_text, add_bos=False)
            token_list = pad_sequences([token_list], maxlen=self._hyperparameters['max_sequence_len'] - 1,
                                       padding='pre', truncating='pre')

            predicted_probs = self._model.predict(token_list)
            predicted_probs = predicted_probs[-1]
            predicted_probs[unk_idx] = np.float32(0.0)
            predicted = np.argmax(predicted_probs)

            output_word = self._tokenizer.de_numericalize([predicted])
            seed_text += "".join(output_word)
        return seed_text


tokenizer = TibetanTokenizer(folder_prefix + 'word_frequencies.json',
                             min_word_frequency=hyperparameters['min_word_frequency'])

lm_trainer = LstmLanguageModelTrainer(hyperparameters, tokenizer)
training, validation = lm_trainer.get_data(corpus_folder + '/**/*.txt',
                                           report_file_load_times=report_file_load_times)

# Try to continue training with a model that was already saved during training
# If that is not possible then create a fresh model
if lm_trainer.load_model(f'{hyperparameters["file_name"]}_train'):
    print('loading existing model')
    lm_trainer._hyperparameters = hyperparameters
else:
    print('starting to train new model')
    lm_trainer.create_model()

# Use learning rate finder?
if lr_find:
    lm_trainer.lr_find(training)

# Continue training the model unless the script is only meant to predict
elif not only_predict:
    lm_trainer.train_model(training, validation)
    lm_trainer.save_model(hyperparameters['file_name'])

# predict
print(lm_trainer.get_model().summary())
print('___')
print(lm_trainer.generate_text("ཕྱི་རོལ་པའི་ཐུབ་པ་དང་སྟོན་པར་གྲགས་པ་དག་དང་", 100))
print('___')
print(lm_trainer.generate_text("སློབ་དཔོན་ ", 300))
