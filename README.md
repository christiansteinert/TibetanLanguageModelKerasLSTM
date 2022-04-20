# Train a statistical language model for Tibetan with Keras, based on LSTM architecture
This repository contains a python based implementation of a machine learning based language model for Tibetan. 
It uses Tensorflow with Keras 2 and an LSTM-based architecture to train a language model over a corpus of Tibetan texts. The corpus must be in Unicode-encoded Tibetan. Tokenization of the corpus is done on the fly as data is loaded by using [Esukhia's botok tokenizer](https://github.com/Esukhia/botok).

The main script (03_train_language_model.py) can be executed as standalone python script or alternatively it can be copied into a single cell of a Jupyter Notebook either locally or on Google Colab.

The model is saved after each epoch and if the code is run again, it will load an existing model and continue training based on the existing model parameters. The code does not go through the entire corpus during each epoch, instead it randomly samples a fraction of the corpus during each epoch in order to keep the epoch duration down to a reasonable duration to allow for regular saving and restart-ability. Despite the random sampling of the corpus, the code still keeps a stable split between test and validation data.

The training was done by using the [BDRC corpus](https://zenodo.org/record/821218) and the [Esukhia Derge Tengyur](https://github.com/Esukhia/derge-tengyur/tree/master/text) as main corpus data with some other stuff sprinkled in.
