

import argparse, sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
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
PLOT_ALL_POINTS = False # Only show points that are a neighbor of a selected word
LABELS_MAP = {} # list of label to point with label -> [x, y, z] for only labeled words
CALLBACK_ADDED = False
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

        

def getNeighbors(distances, indices, wordBidict, word):
    """
    Get the computed neighbors for a given word
    """
    wordInd = wordBidict[word]
    
    # get the row for this word
    wordNeighbors = indices[wordInd]
    wordDistances = distances[wordInd]
    
    neighbors = []
    
    for i in range(0, len(wordNeighbors)):
        neighbor = wordBidict[wordNeighbors[i]]
        distance = wordDistances[i]
        neighbors.append(neighbor)
    
    return neighbors
        


def computeNeighbors(features, n):
    """
    Compute all pairs nearest neighbors
    """
    print ("Computing", n, "nearest neighbors")
    #nbrs = NearestNeighbors(n_neighbors=n, algorithm='brute').fit(features)
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    return distances, indices
    


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
    if not PLOT_ALL_POINTS:
        # iterate LABELS_MAP and get features for words
        xs = np.empty((len(LABELS_MAP),))
        ys = np.empty((len(LABELS_MAP),))
        c = 0
        for w in LABELS_MAP:
            x, y, _ = LABELS_MAP[w]
            xs[c] = x
            ys[c] = y
            c = c + 1
    else:
        xs = features[:,0] # dim 0
        ys = features[:,1] # 1

    AXES.scatter(xs, ys, c='r', marker='o')
    
    for k in LABELS_MAP:
        x, y = LABELS_MAP[k][0:2]
        AXES.annotate(k, xy=(x, y), xytext=(x, y))
                 
    AXES.set_xlabel('X='+str(0))
    AXES.set_ylabel('Y='+str(1))
            
    plt.show()
    plt.draw()
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Analyze Features')
    
    parser.add_argument('--features', help='file containing features')
    args = parser.parse_args()
    
    featuresFile = args.features
    if (featuresFile == None):
        print ("Must supply features file")
        sys.exit(1)
        
    plt.ion() # set interactive mode on so non-blocking
    wordBidict, features, DATA_DIMS = readFeatures(featuresFile)
    distances, indices = computeNeighbors(features, NUM_NEIGHBORS)
    #plotFeatures(features, wordBidict, CURRENT_DIMS)
    
    
    while True:
        c = int(input("add words (1), remove words (2), add all (3), EXIT (-1): "))
        if (c == -1):
            sys.exit(0)
            
        elif (c == 1):
            text = input("Enter comma delimited words to show: ")
            words = text.split(',')
            show = input("Show neighbors? y/n: ") == 'y'

            toShow = set()
            [toShow.add(w) for w in words]

            if show:
                neighbors = []
                for w in words:
                    ns = getNeighbors(distances, indices, wordBidict, w)
                    neighbors.extend(ns)
                    print ("neighbors to", w, " = ", ns)
                for n in neighbors:
                    toShow.add(n)

            # add each word for the LABELS_MAP if not already added
            for w in toShow:
                if not w in LABELS_MAP:
                    x = features[wordBidict[w], 0]
                    y = features[wordBidict[w], 1]
                    LABELS_MAP[w] = [x, y, None] # none is placeholder for annotation
            
            plotFeatures(features, wordBidict)
        
        elif (c == 2):
            text = input("Remove label for words: ")
            words = text.split(',')
            for w in words:
                if w in LABELS_MAP:
                    del LABELS_MAP[w]
            plotFeatures(features, wordBidict) 
        
        elif (c == 3):
            print ("Adding all words to plot")
            for w in VOCAB:
                if not w in LABELS_MAP:
                    x = features[wordBidict[w], 0]
                    y = features[wordBidict[w], 1]
                    LABELS_MAP[w] = [x, y, None] # none is placeholder for annotation
            
            plotFeatures(features, wordBidict)
            
            
            
