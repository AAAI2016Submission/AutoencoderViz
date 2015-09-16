

import argparse, sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import pylab

# from http://stackoverflow.com/questions/1456373/two-way-reverse-map
class bidict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2

        
   

DATA_DIMS = None # num words by num feats per word
FIG = None
AXES = None
NUM_NEIGHBORS = 5
LABELS_MAP = {} # list of label to point with label -> [x, y, z] for only labeled words
VOCAB = set()
        
def readFeatures(featuresFile):
    """
    Load all features from file and populate features array and bidict
    bidict[word] maps to index of word feature in features array
    """
    print ("Reading features")
    global VOCAB
    with open(featuresFile, 'r') as f:
        line = f.readline()
        dims = [int(x) for x in line.split()]
        
        features = np.empty(dims)
        wordBidict = bidict()
        
        wordCount = 0
        for line in f:
            wordVecPair = line.strip().split(' ') # split word and vector
            word = wordVecPair[0]
            feature = wordVecPair[1].split(',')
            
            VOCAB.add(word)
            
            # map the word to its index in the array
            # and the index in the array to the word
            wordBidict[word] = wordCount
            
            for i in range(len(feature)):
                val = feature[i]
                features[wordCount][i] = val
            
            wordCount += 1
            
        return wordBidict, features, dims

        

def computeTSNE(features):
    """
    Compute TSNE for all feature vectors reducing to 2 dims
    
    features is an numpy array
    """
    print ("Computing TSNE for output dim 2")
    
    model = TSNE(n_components=2)
    transformedFeatures = model.fit_transform(features)
    return transformedFeatures
    


def updateLabelsForDims():
    """ 
    When the dimensions are updated the position of the label
    for each word also needs to be updated for the new dimensions
    """
    #print ("CUR_DIMS=", CURRENT_DIMS)
    # when the dimensions are updated, the word position needs to be updated too
    global LABELS_MAP
    for w in LABELS_MAP:
        #print ("LABELS MAP [", w,"] = ", LABELS_MAP[w])
        x = features[wordBidict[w], 0]
        y = features[wordBidict[w], 1]
        # don't want to assign array because could have already created annotation
        LABELS_MAP[w][0] = x 
        LABELS_MAP[w][1] = y 
        #print ("LABELS MAP [", w,"] = ", LABELS_MAP[w])



def setupPlot():
    """
    Initialize a global figure and axes that will be used for drawing
    """
    global FIG
    global AXES
    FIG, AXES = plt.subplots()
    
    

# http://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
def plotFeatures(features, wordBidict):
    """ 
    Plot the desired points. Will draw all points if that is requested. Will draw labels
    for added words. 
    """
    if (FIG == None and AXES == None):
        setupPlot()

    AXES.clear()
    
    xs, ys = None, None

    # iterate LABELS_MAP and get features for words
    xs = np.empty((len(LABELS_MAP),))
    ys = np.empty((len(LABELS_MAP),))
    c = 0
    for w in LABELS_MAP:
        x, y, _ = LABELS_MAP[w]
        xs[c] = x
        ys[c] = y
        c = c + 1

    AXES.scatter(xs, ys, c='r', marker='o')
    
    for k in LABELS_MAP:
        x, y = LABELS_MAP[k][0:2]
        AXES.annotate(k, xy=(x, y), xytext=(x, y))
                 
    AXES.set_xlabel('X='+str(0))
    AXES.set_ylabel('Y='+str(1))
            
    plt.show()
    plt.draw()
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Analyze Features TSNE')
    
    parser.add_argument('--features', help='file containing features')
    args = parser.parse_args()
    
    featuresFile = args.features
    if (featuresFile == None):
        print ("Must supply features file")
        sys.exit(1)
        
    plt.ion() # set interactive mode on so non-blocking
    wordBidict, features, DATA_DIMS = readFeatures(featuresFile)
    transformedFeatures = computeTSNE(features)
    
    
    ## to test just show all
    print ("Adding all words to plot")
    for w in VOCAB:
        if not w in LABELS_MAP:
            x = features[wordBidict[w], 0]
            y = features[wordBidict[w], 1]
            LABELS_MAP[w] = [x, y, None] # none is placeholder for annotation

    plotFeatures(transformedFeatures, wordBidict)
    
    input("Close [enter]")
    
    
    
