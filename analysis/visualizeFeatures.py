

import argparse, sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
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
CURRENT_DIMS = [0, 1, 2]
CALLBACK_ADDED = False
VOCAB = set()
        
def readFeatures(featuresFile):
    """
    Load all features from file and populate features array and bidict
    bidict[word] maps to index of word feature in features array
    """
    print ("Reading features")
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
        x = features[wordBidict[w],CURRENT_DIMS[0]]
        y = features[wordBidict[w],CURRENT_DIMS[1]]
        z = features[wordBidict[w],CURRENT_DIMS[2]]
        # don't want to assign array because could have already created annotation
        LABELS_MAP[w][0] = x 
        LABELS_MAP[w][1] = y 
        LABELS_MAP[w][2] = z
        #print ("LABELS MAP [", w,"] = ", LABELS_MAP[w])



def setupPlot():
    """
    Initialize a global figure and axes that will be used for drawing
    """
    global FIG
    global AXES
    FIG = plt.figure()
    AXES = FIG.add_subplot(111, projection='3d')

    

# http://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
def plotFeatures(features, wordBidict, dims):
    """ 
    Plot the desired points. Will draw all points if that is requested. Will draw labels
    for added words. 
    """
    if (FIG == None and AXES == None):
        setupPlot()

    AXES.clear()
    
    xs, ys, zs = None, None, None
    if not PLOT_ALL_POINTS:
        # iterate LABELS_MAP and get features for words
        xs = np.empty((len(LABELS_MAP),))
        ys = np.empty((len(LABELS_MAP),))
        zs = np.empty((len(LABELS_MAP),))
        c = 0
        for w in LABELS_MAP:
            x, y, z, _ = LABELS_MAP[w]
            xs[c] = x
            ys[c] = y
            zs[c] = z
            c = c + 1
    else:
        xs = features[:,dims[0]] # dim 0
        ys = features[:,dims[1]] # 1
        zs = features[:,dims[2]] # 2

    AXES.scatter(xs, ys, zs, c='r', marker='o')
    
    for k in LABELS_MAP:
        x1, y1, z1 = LABELS_MAP[k][0:3]
        x2, y2, _ = proj3d.proj_transform(x1, y1, z1, AXES.get_proj())
        annotation = pylab.annotate(k, xy=(x2,y2), textcoords="offset points", xytext=(0, 0))
        #annotation = pylab.annotate(k, xy = (x2, y2), xytext = (-15, 15), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        
        LABELS_MAP[k][3] = annotation
                 
    AXES.set_xlabel('X='+str(dims[0]))
    AXES.set_ylabel('Y='+str(dims[1]))
    AXES.set_zlabel('Z='+str(dims[2]))
    
    global CALLBACK_ADDED
    if not CALLBACK_ADDED:
        FIG.canvas.mpl_connect('button_release_event', update_position)
        #FIG.canvas.mpl_connect('motion_notify_event', update_position)
        CALLBACK_ADDED = True
        
    plt.show()
    plt.draw()
    
    
    
def update_position(e):
    """
    A callback handler to update positions of the labels projected into 2d space
    when the graph is rotated
    """
    for k in LABELS_MAP:
        x1, y1, z1 = LABELS_MAP[k][0:3]
        x2, y2, _ = proj3d.proj_transform(x1, y1, z1, AXES.get_proj())
        annotation = LABELS_MAP[k][3]
        annotation.xy = x2, y2
        annotation.update_positions(FIG.canvas.renderer)
    FIG.canvas.draw()

        
        

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
        c = int(input("Enter option dims (1), add words (2), remove words (3), add all (4), EXIT (-1): "))
        if (c == -1):
            sys.exit(0)
            
        elif (c == 1):
            dims = input("Enter 3 comma separated dims (max="+str(DATA_DIMS[1]-1)+"): ")
            dims = [int(x) for x in dims.split(',')]
            # check that there are 3 dims and are in range
            if len(dims) != 3:
                print ("Can only enter 3 dims")
                continue
            for d in dims:
                if d < 0 or d > DATA_DIMS[1]-1:
                    print ("Dimension [" + str(d) + "] out of range")
                    continue          
            CURRENT_DIMS = dims
            updateLabelsForDims()
            plotFeatures(features, wordBidict, CURRENT_DIMS)
            
        elif (c == 2):
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
                    x = features[wordBidict[w],CURRENT_DIMS[0]]
                    y = features[wordBidict[w],CURRENT_DIMS[1]]
                    z = features[wordBidict[w],CURRENT_DIMS[2]]
                    LABELS_MAP[w] = [x, y, z, None] # none is placeholder for annotation
            
            plotFeatures(features, wordBidict, CURRENT_DIMS)
        
        elif (c == 3):
            text = input("Remove label for words: ")
            words = text.split(',')
            for w in words:
                if w in LABELS_MAP:
                    del LABELS_MAP[w]
            plotFeatures(features, wordBidict, CURRENT_DIMS)
            
        elif (c == 4):
            print ("Adding all words to plot")
            for w in VOCAB:
                if not w in LABELS_MAP:
                    x = features[wordBidict[w],CURRENT_DIMS[0]]
                    y = features[wordBidict[w],CURRENT_DIMS[1]]
                    z = features[wordBidict[w],CURRENT_DIMS[2]]
                    LABELS_MAP[w] = [x, y, z, None] # none is placeholder for annotation
            
            plotFeatures(features, wordBidict, CURRENT_DIMS)         

