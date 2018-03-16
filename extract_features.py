import os
import pickle

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input

def extract_features(directory = 'data'):
    """
    Goes into directory, finds the images, and feeds into VGG16 model to compute features

    """
    # load model
    in_layer = Input(shape=(224,224,3))
    model = VGG16(include_top = False, input_tensor = in_layer)
    # print(model.summary())
    # store features from each image in a dict
    features = {}
    # traverse directories
    for (root, dirnames, filenames) in os.walk(directory):
        # go into each directory and get the images
        for d in dirnames:
            path = os.path.join(root,d)
            for (_,_, filenames) in os.walk(path):
                for f in filenames:
                    f = os.path.join(path,f)
                    if f.endswith('.png') or f.endswith('.jpg'):
                        # load img.
                        # TODO: Use grayscale because ink color and paper color doesn't matter for the structure, but VGG needs to have three channel input...need to figure that out.
                        # Some people just duplicate the 224,224,1 three times
                        img = load_img(f, target_size = (224,224), grayscale = False)
                        # convert img pixels to numpy array
                        img = img_to_array(img)
                        # reshape data for model
                        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                        # prepare image for VGG model
                        img = preprocess_input(img)
                        # get features
                        feature = model.predict(img, verbose = 1)
                        # get image id to use as the key in features dict
                        # basename returns just the last part of the path
                        # returns the part before .png
                        img_id = os.path.basename(os.path.normpath(f.split('.')[0]))
                        # store feature
                        features[img_id] = feature
                        print('>%s' % img_id)
    return features

if __name__ == '__main__':
    """ if this script is run, it will extract features for the photos via VGG16"""
    directory = 'data'
    features = extract_features(directory)
    print('Finished extracting features --- # of features: ', len(features))
    print('Saving to features.pkl')
    pickle.dump(features, open('features.pkl', 'wb'))

