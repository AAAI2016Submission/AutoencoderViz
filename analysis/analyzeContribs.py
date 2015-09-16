

from operator import itemgetter
import argparse, sys

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
    print ("Reading contribs")
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
            contribsMap[int(hiddenInd)] = sorted(contribs, key=itemgetter(1), reverse=True)
        
        return contribsMap
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Analyze Contribs')
    
    parser.add_argument('--worddict', help='file with word mapping')
    parser.add_argument('--contribs', help='file containing contribs for each hidden node')
    args = parser.parse_args()
    
    contribsFile = args.contribs
    wordDictFile = args.worddict
    if (contribsFile == None or wordDictFile == None):
        print ("Must supply contribs file")
        sys.exit(1)
        
    wordBidict = readWordDict(wordDictFile)
    contribs = readContribs(contribsFile, wordBidict)

    #for k in contribs:
    #    print (k, contribs[k])
        
    while True:
        maxVal = len(contribs)-1
        hidden = int(input("Select hidden node to view (max " + str(maxVal) + "): "))
        
        if (hidden <= maxVal):
            print (contribs[hidden])




