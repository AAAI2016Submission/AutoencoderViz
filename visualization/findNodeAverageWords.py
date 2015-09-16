


# load contribs
# read word2vec output
# for each node
#    for each word in first k contribs
#       create avg vector using linear combination
#    find nearest word in word2vec output



import os, argparse, sys, math, operator
import numpy
from sklearn.preprocessing import normalize

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
        
        

def readWordDict(wordDictFile):
    wordBidict = bidict()
    with open(wordDictFile, 'r') as f:
        for line in f:
            line = line.strip()
            word, indStr = line.split()
            #print (word, indStr)
            wordBidict[word] = int(indStr)

    return wordBidict
        
    
    
def readContribs(contribsFile, wordBidict):
    print ("Reading contribs:", contribsFile)
    with open(contribsFile, 'r') as f:
        contribsMap = {}
        for line in f:
            line = line.strip()
            hiddenInd, contribsStr = line.split()
            contribPairs = contribsStr.split(",")
            
            contribs = []
            for i in range(len(contribPairs)):
                pair = contribPairs[i]
                key, value = pair.split(":")
                contribs.append((wordBidict[int(key)], float(value)))
            contribsMap[int(hiddenInd)] = sorted(contribs, key=operator.itemgetter(1), reverse=True)
        
        return contribsMap
        

# can do linear sort since know where should be
def sortFiles(files):
    sortedFiles = [None]*len(files)
    for f in files:
        n = int(f[0:-8]) # find the itr number
        sortedFiles[n-1] = f    
    return sortedFiles
        
    

def findContribFiles(contribsDir):
    files = os.listdir(contribsDir)
    try:
        files.remove("default.txt") # don't want to include this one
    except:
        print ("Removing default.txt")
        
    sortedFiles = sortFiles(files)
    return sortedFiles


def loadWord2VecData(path):
    with open(path, 'r') as f:
        header = f.readline()
        header = header.split(" ")
        dims = (int(header[0]), int(header[1]))
        
        word2VecBidict = bidict()
        word2VecFeatures = []
        #features = numpy.empty(dims)
        
        count = 0
        for line in f:
            line = line.strip().split(" ")
            word = line[0]
            vec = line[1:]
            vec = [float(v) for v in vec]
            
            word2VecFeatures.append(vec)
            #for i in range(len(vec)):
            #    features[count][i] = vec[i]
            
            word2VecBidict[word] = count
            count += 1
            
    return word2VecFeatures, word2VecBidict
                    


def normalizeContrib(contrib, normMin=0, normMax=1):
    # this expects contribs to be ordered high to low. Currently it is always that way
    #print (contrib)
    dataMin = contrib[len(contrib)-1][1]
    dataMax = contrib[0][1]
    
    #print (dataMin, dataMax)
    
    for c in range(len(contrib)):
        pair = contrib[c]
        word = pair[0]
        val = pair[1]
        newVal = ( (normMax - normMin) * (val - dataMin) ) / ( dataMax - dataMin ) + normMin
        contrib[c] = (word, newVal)
    
    # print (contrib)
    return contrib            




def getWeightedVector(contrib, word2VecFeatures, word2VecBidict, k=5):
    totalVec = None
    
    n = min(k, len(contrib))
    
    # contrib should be sorted high to low
    maxVal = abs(contrib[0][1])
    minVal = abs(contrib[-1][1])
    
    newContrib = []
    if (maxVal > minVal): 
        # use front
        for i in range(len(contrib)):
            newContrib.append(contrib[i])
    else:
        # use back
        for i in range(len(contrib) - 1, -1, -1):
            newContrib.append(contrib[i])
        
        # if the most significant value was negative
        # swap the sign of all values in contrib
        for j in range(len(newContrib)):
            c = newContrib[j]
            newContrib[j] = (c[0], -1*c[1])
    
    #print ("new contrib = ", newContrib)
    
    contrib = normalizeContrib(newContrib)
       
    words = []
    normVals = []
    
    for i in range(n):
        word, val = contrib[i]
        words.append(word)
        normVals.append(val)
        
    #print (words)
    #print (normVals)
    
    for i in range(n):
        word = words[i]
        wordVec = word2VecFeatures[word2VecBidict[word]]
        #print (word)
        #return wordVec
        wordVec = [normVals[i] * x for x in wordVec]
    
        if (totalVec == None):
            totalVec = wordVec
        else:
            for i in range(len(totalVec)):
                totalVec[i] += wordVec[i]
                
    totalVec = [ float(x) / n for x in totalVec ]

    return totalVec        


def euclideanDist(a, b):
    s = 0.0
    for i in range(len(a)):
        s += math.pow( (a[i] - b[i]), 2.0)
    return math.sqrt(s)
    

def findNearestVector(weightedVec, word2VecFeatures, word2VecBidict, k=25):    
    wordDists = []
    
    for i in range(len(word2VecFeatures)):
        vec = word2VecFeatures[i]
        dist = euclideanDist(vec, weightedVec)
        
        wordDists.append((word2VecBidict[i], dist))
    
    # sort word dists 
    wordDists.sort(key=operator.itemgetter(1))
    return wordDists[0:k]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Word Cloud from contribs')
    
    parser.add_argument('--worddict', help='file with word mapping')
    parser.add_argument('--contribsdirl1', help='directory containing contribs for each hidden node')
    parser.add_argument('--contribsdirl2', help='directory containing contribs for each hidden node')
    parser.add_argument('--word2vecpath', help='directory containing contribs for each hidden node')
    #parser.add_argument('--word2vecpath2', help='directory containing contribs for each hidden node')
    
    args = parser.parse_args()
    
    contribsDirL1 = args.contribsdirl1
    contribsDirL2 = args.contribsdirl2  
    wordDictFile = args.worddict
    word2VecPath = args.word2vecpath
    #word2VecPath2 = args.word2vecpath2
    
    if (contribsDirL1 == None or contribsDirL2 == None or wordDictFile == None):
        print ("Must supply contrib dir and word dict and imageDir")
        sys.exit(1)
    
    # load data
    wordBidict = readWordDict(wordDictFile)
    sortedFilesL1 = findContribFiles(contribsDirL1)
    sortedFilesL2 = findContribFiles(contribsDirL2)
    word2VecFeatures, word2VecBidict = loadWord2VecData(word2VecPath)
    #word2VecFeatures2, word2VecBidict2 = loadWord2VecData(word2VecPath2)
    
    numFilesL1 = len(sortedFilesL1)
    numFilesL2 = len(sortedFilesL2)
    
    numNodesL1 = len(readContribs(os.path.join(contribsDirL1, sortedFilesL1[0]), wordBidict))
    numNodesL2 = len(readContribs(os.path.join(contribsDirL2, sortedFilesL2[0]), wordBidict))
    
    while True:
        itrNum = int(input("Choose itr number (1-indexed): ")) - 1
        nodeNum = int(input("Choose node number (1-indexed): ")) - 1
        
        contribDir = None
        contribFile = None
        
        if (itrNum < numFilesL1):
            contribDir = contribsDirL1
            contribFile = sortedFilesL1[itrNum]
        elif (itrNum < numFilesL1 + numFilesL1):
            contribDir = contribsDirL2
            contribFile = sortedFilesL2[itrNum - numFilesL1]
        
        path = os.path.join(contribDir, contribFile)
        contribs = readContribs(path, wordBidict)
        
        #print (contribs)
        weightedVec = getWeightedVector(contribs[nodeNum], word2VecFeatures, word2VecBidict) 
        wordDists = findNearestVector(weightedVec, word2VecFeatures, word2VecBidict)
#        wordDists = findNearestVector(weightedVec, word2VecFeatures2, word2VecBidict2)

        print (wordDists)
        
        
        
