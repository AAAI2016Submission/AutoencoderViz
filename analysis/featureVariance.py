
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
        
        

def dimsByVariance(dims, wordBidict, features, words):
    """ 
    Return a list of tupes of (variance, dimension) ordered by least to greatest variance
    """
    vecs = []
    for w in words:
        wvec = features[wordBidict[w]]
        vecs.append(wvec)

    dimVars = [0]*dims[1]
    for d in range(dims[1]):
        minVal = None
        maxVal = None
        for v in range(len(vecs)):
            if minVal == None or vecs[v][d] < minVal:
                minVal = vecs[v][d]
            if maxVal == None or vecs[v][d] > maxVal:
                maxVal = vecs[v][d]
        dimVars[d] = maxVal - minVal

    dimVarsPairs = [(dimVars[i], i) for i in range(len(dimVars))]
    sortedPairs = sorted(dimVarsPairs)
    
    return sortedPairs
    


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
            
        return dims, wordBidict, features
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Analyze Features')
    
    parser.add_argument('--features', help='file containing features')
    args = parser.parse_args()
    
    featuresFile = args.features
    if (featuresFile == None):
        print ("Must supply features file")
        sys.exit(1)
        
    dims, wordBidict, features = readFeatures(featuresFile)
    
    while True:
        text = input("Enter comma delimited set of words to analyze: ")
        words = text.split(',')
        dimVars = dimsByVariance(dims, wordBidict, features, words)
        print (words)
        print (dimVars)
        print ()


