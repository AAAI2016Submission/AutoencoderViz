

import argparse, sys
import numpy

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
        

def findNeighbors(distances, indices, wordBidict, word):
    print ("Finding neighbors for", word)
    wordInd = wordBidict[word]
    
    # get the row for this word
    wordNeighbors = indices[wordInd]
    wordDistances = distances[wordInd]
    
    print ("Nearest words to:", word)
    
    for i in range(len(wordNeighbors)):
        neighbor = wordBidict[wordNeighbors[i]]
        distance = wordDistances[i]
        print (neighbor, "at distance", distance)
    
    print()
        

def computeNeighbors(features, n):
    print ("Computing", n, "nearest neighbors")
    #nbrs = NearestNeighbors(n_neighbors=n, algorithm='brute').fit(features)
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    return distances, indices
    
    
def readFeatures(featuresFile):
    print ("Reading features")
    with open(featuresFile, 'r') as f:
        line = f.readline()
        dims = [int(x) for x in line.split()]
        
        features = numpy.empty(dims)
        wordBidict = bidict()
        
        wordCount = 0
        for line in f:
            wordVecPair = line.strip().split(' ') # split word and vector
            word = wordVecPair[0]
            feature = wordVecPair[1].split(',')
            
            # map the word to its index in the array
            # and the index in the array to the word
            wordBidict[word] = wordCount
            
            for i in range(len(feature)):
                val = feature[i]
                features[wordCount][i] = val
            
            wordCount += 1
            
        return wordBidict, features
        
 
def readCentroids(centroidsFile):
    print ("Reading centroids")
    with open(centroidsFile, 'r') as f:
        centroids = []
        for line in f:
            line = line.strip()
            centroidWords = line.split(",")
            centroids.append(centroidWords)
            
        return centroids

        
def computeCentroids(centroidsWords, features, wordBidict):
    centroids = []
    
    for c in centroidsWords:
        cenSum = 0
        for w in c:
            # get the feature
            cenSum += features[wordBidict[w]]
        centroids.append(cenSum / len(c))
        
    return centroids
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Analyze Centroids')
    
    parser.add_argument('--features', help='file containing features')
    parser.add_argument('--centroids', help='file containing centroids')

    args = parser.parse_args()
    
    featuresFile = args.features
    if (featuresFile == None):
        print ("Must supply features file")
        sys.exit(1)
        
    wordBidict, features = readFeatures(featuresFile)
    
    # read centroid words
    centroidsWords = readCentroids(args.centroids)
    centroids = computeCentroids(centroidsWords, features, wordBidict)
    
    print ("Centroids = ", centroids)
    # compute inter / intra cluster distances
    
    
    
    
    
