# read in a file of vectors where I know the ith vector
# is the hidden features from the vector with 1 in the ith position

import argparse

indToWordDict = {}

def loadWordDict(fname):
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            word, index = line.split()
            indToWordDict[int(index)] = word


def readVecsAndWriteFeatures(fnameVectors, fnameFeatures):
    with open(fnameVectors, 'r') as fvectors:
        with open(fnameFeatures, 'w') as ffeatures:
            c = 0
            for line in fvectors:
                line = line.strip()
                vec = line.replace(' ', ',')
                ffeatures.write(indToWordDict[c] + ' ' + vec + '\n')
                c = c + 1
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Vec to Features')
    
    parser.add_argument('--worddict', help='file of word to index')
    parser.add_argument('--vectors', help='vectors file')
    parser.add_argument('--features', help='feature output')
    
    args = parser.parse_args()
    
    loadWordDict(args.worddict)
    readVecsAndWriteFeatures(args.vectors, args.features)
