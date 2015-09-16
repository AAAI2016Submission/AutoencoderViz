

from os import path
from operator import itemgetter
import argparse, sys
import matplotlib.pyplot as plt
from scipy.misc import imread
from wordcloud import WordCloud


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
        

wordColorMap = {}

def normalizeContrib(contrib, normMin=0, normMax=1):
    # this expects contribs to be ordered high to low. Currently it is always that way
    dataMin = contrib[len(contrib)-1][1]
    dataMax = contrib[0][1]
    
    for c in range(len(contrib)):
        pair = contrib[c]
        word = pair[0]
        val = pair[1]
        newVal = ( (normMax - normMin) * (val - dataMin) ) / ( dataMax - dataMin ) + normMin
        contrib[c] = (word, newVal)
        
        # add word to color map
        wordColorMap[word] = 1 if (newVal > 0) else -1
        
    return contrib            


def generateText(contrib, k=None):
    # gen text for first k words, all words if k == None
    text = ""
    if (k == None):
        k = len(contrib)
        
    for i in range(k):
        pair = contrib[i]
        word = pair[0]
        val = pair[1]
        num = round(abs(val) * 100)
        text += (word + ' ') * num
    
    return text
        
        

def pos_neg_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    #print ("word = ", word, wordColorMap[word])
    if (wordColorMap[word] == 1):
        return "rgb(0, 255, 0)"
    elif (wordColorMap[word] == -1):
        return "rgb(0, 0, 255)"
    
     

# contrib is an array of word, value pairs
def generateWordCloud(node, contribs, wordsToShow=None, normalize=True, normMin=0, normMax=1):
    contrib = contribs[node]
    
    if (normalize):
        contrib = normalizeContrib(contrib, normMin, normMax)
    
    # generate text
    text = generateText(contrib, wordsToShow)
    
    # load mask
    d = path.dirname(__file__)
    circle_mask = imread(path.join(d, "black_circle_mask_whitebg.png"))
    
    # gen word cloud
    wc = WordCloud(background_color="white", max_words=2000, mask=circle_mask)
    wc.generate(text)

    # store to file
    wc.to_file(path.join(d, "node.png"))

    # show
    useColorFunc = False #normalize
    if (useColorFunc):
        plt.imshow(wc.recolor( color_func=pos_neg_color_func ))
    else:
        plt.imshow(wc)
        
    plt.axis("off")
    plt.show()
    
             

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Word Cloud from contribs')
    
    parser.add_argument('--worddict', help='file with word mapping')
    parser.add_argument('--contribs', help='file containing contribs for each hidden node')
    parser.add_argument('--maxwords', help='number of input nodes to consider in order of most to least significant')
    parser.add_argument('--normmin', default=0, help='min val in normalization')
    parser.add_argument('--normmax', default=1, help='max val in normalization')
    
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.set_defaults(normalize=False)
    
    args = parser.parse_args()
    
    contribsFile = args.contribs
    wordDictFile = args.worddict
    if (contribsFile == None or wordDictFile == None):
        print ("Must supply contribs file")
        sys.exit(1)
        
    wordBidict = readWordDict(wordDictFile)
    contribs = readContribs(contribsFile, wordBidict)
        
    maxwords = None if args.maxwords == None else int(args.maxwords)
    normMin = int(args.normmin)
    normMax = int(args.normmax)
    normalize = bool(args.normalize)
    
    while True:
        maxVal = len(contribs)-1
        hidden = int(input("Select hidden node to generate wordcloud for (max " + str(maxVal) + "): "))
        
        if (hidden <= maxVal):
            generateWordCloud(hidden, contribs, maxwords, normalize, normMin, normMax)
            
            
