import os
import pickle
import numpy as np

from nltk.translate.bleu_score import corpus_bleu

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D

def load_smiles(filename):
    # load the SMILES text
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_image_smiles(directory):
    """
    Prepare image with its target (SMILES)
    {photo_id: smiles}

   Arguments: directory. directory with training data

    Returns:
    mapping - dict with photo_id as key and smiles as value

    """
    mapping = dict()
    for (root, dirnames, filenames) in os.walk(directory): # os.walk returns dirnames and filenames as lists
        # go into each directory
        for d in dirnames:
            # '/data/d'
            path = os.path.join(root,d)
            # first get the smiles information, or else it say smiles referenced before assignment
            # all the images in a given directory are the same molecule, so they have the same smiles
            smiles = load_smiles(os.path.join(path, 'smiles.txt')).strip('\n')
            # then look inside the directory for the image files
            for (_, _, filenames) in os.walk(path):
                for f in filenames:
                    if f.endswith('.png') or f.endswith('.jpg'):
                        # don't include .png in dictionary key
                        mapping[f[:-4]] = smiles
    return mapping

def train_test_split(mapping, train_size, test_size):
    """
    Might want to replace with sklearn train test split, but just use this for now for baseline model
    Arguments:
    mapping: dict {photo_id: smiles}
    train_size: number of training images
    test_size: number of test images

    Returns:
    two sets, train and test

    """
    dataset = mapping.keys()
    # order the keys so the split is consistent
    ordered = sorted(dataset)
    # return split data set as two new sets
    return set(ordered[:train_size]), set(ordered[train_size:train_size + test_size])

def load_clean_smiles(mapping, dataset):
    """
    Adds start and end tokens to the smiles for the photo_id's in the dataset

    Arguments:
    ----------
    mapping: dict {photo_id:smiles}
    dataset: either train or test. set of photo_id

    Returns:
    --------
    dict of cleaned smiles with start and end tokens for the photo_id that are in dataset

    """
    cleaned_smiles = dict()
    for photo_id in dataset:
        # store the tokens with start and end tokens
        # "!" is the startsequence and "$" is the endsequence
        cleaned_smiles[photo_id] = '!' + mapping[photo_id] + '$'
    return cleaned_smiles

def load_image_features(pickled_file, cleaned_smiles):
    """
    Load pre-computed features for the photos that are in the dataset

    Arguments:
    ----------
    pickled_file: the pre computed features. opened up will be a dict
    cleaned_smiles: a set, not a dict

    Returns:
    --------
    dict of {photo_id, features} for the selected images in the data sets
    """
    # load all features
    all_features = pickle.load(open(pickled_file, 'rb'))
    # select only features for the photos in the data sets
    features = {k: all_features[k] for k in cleaned_smiles}
    return features


def create_tokenizer(cleaned_smiles):
    """
    Fit keras tokenizer on the cleaned smiles
    Arguments:
    ----------
    cleaned_smiles: dict of photo_id and smiles with !,$ in the dataset

    Returns:
    --------
    keras tokenizer
    """
    smiles = list(cleaned_smiles.values())
    # turn filters off because this is not normal text, it's smiles strings so we want all the symbols etc
    # char_level true because want to split at every character
    tokenizer = Tokenizer(filters = None, char_level = True)
    tokenizer.fit_on_texts(smiles)
    # print(tokenizer.word_counts)
    return tokenizer

def create_sequences(tokenizer, smiles, image, max_length):
    """

    Arguments:
    ----------
    tokenizer: keras tokenizer fitted on the cleaned_smiles
    smiles: STRING of cleaned smiles
    image: photo_id?
    max_length: max_length to allow the model to build seq

    Returns:
    --------
    list of lists of X_images, X_seq, y

    """
    X_images, X_seq, y = [],[],[]
    vocab_size = len(tokenizer.word_index) + 1
    # turns lists of texts into sequences, list of word indexes where the word of rank i in the dataset (starting at 1) has index i
    seq = tokenizer.texts_to_sequences([smiles])[0]
    # split one sequence into multiple X,y pairs
    for i in range(1, len(seq)):
        # select
        in_seq, out_seq = seq[:i], seq[i]
        # pad input seq
        in_seq = pad_sequences([in_seq], maxlen= max_length)[0]
        # print(in_seq)
        # encode output sequence to categoricals
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # store
        X_images.append(image)
        X_seq.append(in_seq)
        y.append(out_seq)

    return [X_images, X_seq, y]

def define_model(vocab_size, max_length):
    """
    The model to generate SMILES strings

    """
    # feature extractor (encoder)
    # features from VGG16 model will be of shape (7,7,512)
    inputs1 = Input(shape=(7,7,512))
    fe1 = GlobalMaxPooling2D()(inputs1)
    fe2 = Dense(128, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe2)
    # embedding
    inputs2 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size, 50, mask_zero = True)(inputs2)
    emb3 = LSTM(256, return_sequences=True)(emb2)
    emb4 = TimeDistributed(Dense(128, activation = 'relu'))(emb3)
    # merge inputs
    merged = concatenate([fe3, emb4])
    # language model
    lm2 = LSTM(500)(merged)
    lm3 = Dense(500, activation='relu')(lm2)
    outputs = Dense(vocab_size, activation = 'softmax')(lm3)

    model = Model(inputs=[inputs1,inputs2], outputs = outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    print(model.summary())

    return model

def data_generator(cleaned_smiles, img_features, tokenizer, max_length, n_step):
    """

    Arguments:
    ----------
    cleaned_smiles: dictionary of cleaned_smiles for the photo_ids in the dataset
    n_step: number of images to generate for each batch
    """
    # loop until training is done
    while 1:
        # loop over photo_id in dataset
        keys = list(cleaned_smiles.keys())
        for i in range(0, len(keys), n_step):
            X_images, X_seq, y = [], [], []
            for j in range(i, min(len(keys), i+n_step)):
                photo_id = keys[j]
                # retrieve the photo from the pickled extracted features dict
                image = img_features[photo_id][0]
                # retrieve the smiles corresponding to that image
                smiles = cleaned_smiles[photo_id]
                # generate input-output pairs
                in_img, in_seq, out_smiles = create_sequences(tokenizer, smiles, image, max_length)
                for k in range(len(in_img)):
                    X_images.append(in_img[k])
                    X_seq.append(in_seq[k])
                    y.append(out_smiles[k])
            # yield this batch of samples to the model
            yield [[np.array(X_images), np.array(X_seq)], np.array(y)]

def char_for_id(integer, tokenizer):
    """
    Map an integer to a character. Used in generate_smiles

    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_smiles(model, tokenizer, photo_id, max_length):
    """
    Generate a smiles for an image

    """
    # start the generation process with the start token "!"
    in_text = '!'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen = max_length)
        # predict next char
        yhat = model.predict([photo_id, sequence], verbose = 0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        char = char_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if char is None:
            break
        # append as input for generating the next word
        in_text += ' ' + char
        # stop if end of sequence is predicted, '$'
        if char == '$':
            break
    return in_text

def evaluate_model(model, smiles, photos, tokenizer, max_length):
    """
    Evaluate the model with BLEU score

    """
    actual, pred = [], []
    # step over the whole set
    for key, smi in smiles.items():
        # generate smiles
        yhat = generate_smiles(model, tokenizer, photos[key], max_length)
        # store actual and predicted smiles
        actual.append([smi.split()])
        predicted.append(yhat.split())
    # calculate BLEU score
    bleu = corpus_bleu(actual, predicted)

    return bleu

#####################################################
# load dataset
directory = 'data'
mapping = load_image_smiles(directory)

X_train, X_test = train_test_split(mapping, 100, 100)
# process smiles
train_smiles = load_clean_smiles(mapping, X_train)
test_smiles = load_clean_smiles(mapping, X_test)
print('SMILES loaded: train=%d, test=%d' % (len(train_smiles), len(test_smiles)))
# photo features
train_features = load_image_features('features.pkl', X_train)
test_features = load_image_features('features.pkl', X_test)
print('Photos loaded: train=%d, test=%d' % (len(train_features), len(test_features)))
# prepare tokenizer
tokenizer = create_tokenizer(train_smiles)
vocab_size = len(tokenizer.word_index) + 1
print('Vocab size: %d' % vocab_size)
print('Vocab: ', tokenizer.word_counts)
# just set a max seq length for this run
max_length = 20

# define experiment
model_name = 'baseline1'
verbose = 2
n_epochs = 50
n_photos_per_update = 2
n_batches_per_epoch = int(len(X_train)/ n_photos_per_update)
n_repeats = 3

# run experiment
train_results, test_results = [], []
for i in range(n_repeats):
    # define model
    model = define_model(vocab_size, max_length)
    # fit model
    model.fit_generator(data_generator(train_smiles, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch=n_batches_per_epoch, epochs = n_epochs, verbose = verbose)
    # evaluate model on training data
    train_score = evaluate_model(model, train_smiles, train_features, tokenizer, max_length)
    test_score = evaluate_model(model, test_smiles, test_features, tokenizer, max_length)

    train_results.append(train_score)
    test_results.append(test_score)
    print('>%d: train=%f test%f' %((i+1), train_score, test_score))

# ## Tests
# mapping = load_image_smiles('data')
# X_train, X_test = train_test_split(mapping, 10, 10)
# cleaned_smiles = load_clean_smiles(mapping, X_train)
# tokenizer = create_tokenizer(cleaned_smiles)
# # print(create_sequences(tokenizer, cleaned_smiles['struct13_09'], 'struct13_09', max_length = 10))
# model = define_model(5,10)
# model.fit_generator(data_generator(train_smiles, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch = n_batches_per_epoch, epochs = n_epochs, verbose = verbose)
