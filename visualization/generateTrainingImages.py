

import os, argparse, sys, time
import matplotlib.pyplot as plt

from PIL import Image # Pillow image
from operator import itemgetter
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
            contribsMap[int(hiddenInd)] = sorted(contribs, key=itemgetter(1), reverse=True)
        
        return contribsMap
        

wordColorMap = {}

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
        
        # add word to color map
        #wordColorMap[word] = 1 if (newVal > 0) else -1
    
    # print (contrib)
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

def gray_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    #print ("word = ", word, wordColorMap[word])
    gray = wordColorMap[word]
    return "hsl(0, 0%, " + str(gray) + "%)"

    
     

# contrib is an array of word, value pairs
def generateWordCloud(node, contribs, maskImg=None, wordsToShow=100, normalize=True, normMin=0, normMax=1):
    contrib = contribs[node]
    
    # Find side of largest magnitude
    # take k words from that side
    # normalize those words (may need abs val)
    # generate text from those normalized values
    
    # contrib should be sorted high to low
    maxVal = abs(contrib[0][1])
    minVal = abs(contrib[-1][1])
    
    #print (contrib[0][1], contrib[-1][1])
    #print ("max min = ", maxVal, minVal)
    
    newContrib = []
    if (maxVal > minVal): 
        # use front
        
        #newContrib = contrib[0:min(len(contrib), wordsToShow))]
        
        for i in range(min(len(contrib), wordsToShow)):
            newContrib.append(contrib[i])
    else:
        # use back
        for i in range(min(len(contrib), wordsToShow) - 1, -1, -1):
            newContrib.append(contrib[i])
        
        # if the most significant value was negative
        # swap the sign of all values in contrib
        for j in range(len(newContrib)):
            c = newContrib[j]
            newContrib[j] = (c[0], -1*c[1])
    
    #print ("new contrib = ", newContrib)
    
    if (normalize):
        contrib = normalizeContrib(newContrib, normMin, normMax)
    
    # for each value in normalized contrib
    # assign color value
    for c in contrib:
        word, val = c
        # add word to color map
        #wordColorMap[word] = int(round(255*(1-val)))
        wordColorMap[word] = int(round(200*(1-val)))
        
    
    # generate text
    text = generateText(contrib, min(len(contrib), wordsToShow))
    
    # gen word cloud
    #s = time.time()
    wc = WordCloud(background_color="white", max_words=2000, mask=maskImg)
    wc.generate(text)
    wc.recolor(color_func=gray_color_func)
    #e = time.time()
    #print ("word cloud only time = ", (e-s))
    return wc    



# for each contrib file in dir
# read contrib
# generate wordcloud
# paste word cloud for each node into larger image
# write image to output directory


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Word Cloud from contribs')
    
    parser.add_argument('--worddict', help='file with word mapping')
    parser.add_argument('--contribsdirl1', help='directory containing contribs for each hidden node')
    parser.add_argument('--contribsdirl2', help='directory containing contribs for each hidden node')
    parser.add_argument('--contribsdirl3', help='directory containing contribs for each hidden node')
    
    parser.add_argument('--imagedir', help='directory to write generated images to')
    
    parser.add_argument('--maxwords', help='number of input nodes to consider in order of most to least significant')
    parser.add_argument('--normmin', default=0, help='min val in normalization')
    parser.add_argument('--normmax', default=1, help='max val in normalization')
    
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.set_defaults(normalize=False)
    
    args = parser.parse_args()
    
    contribsDirL1 = args.contribsdirl1
    contribsDirL2 = args.contribsdirl2  
    contribsDirL3 = args.contribsdirl3
    wordDictFile = args.worddict
    imageDir = args.imagedir
    
    if (contribsDirL1 == None or wordDictFile == None or imageDir == None):
        print ("Must supply contrib dir and word dict and imageDir")
        sys.exit(1)
        
    # create image directory if not exists
    if not os.path.isdir(imageDir):
        os.mkdir(imageDir)
        
    
    # load data
    wordBidict = readWordDict(wordDictFile)
    sortedFilesL1 = findContribFiles(contribsDirL1)
    
    sortedFilesL2 = None
    if (contribsDirL2 != None):
        sortedFilesL2 = findContribFiles(contribsDirL2)
    
    sortedFilesL3 = None
    if (contribsDirL3 != None):
        sortedFilesL3 = findContribFiles(contribsDirL3)
    
    
    maxwords = None if args.maxwords == None else int(args.maxwords)
    normMin = float(args.normmin)
    normMax = float(args.normmax)
    normalize = bool(args.normalize)
    

    # load mask
    d = os.path.dirname(__file__)
    circle_mask = imread(os.path.join(d, "black_circle_mask_whitebg_100x100.png"))
        
    # assume each node word cloud is is 300 x 300
    wordCloudSize = (100, 100)
    padding = (20, 50)
    margins = (50, 50)
        
    # if start < #contribsL1
    # finish first layer
    # else: ensure final image of l1 is loaded and begin layer 2
    
    finalImgL1 = None
    finalImgL2 = None
    
    numFilesL1 = len(sortedFilesL1)
    numFilesL2 = None
    if (sortedFilesL2 != None):
        numFilesL2 = len(sortedFilesL2)
    numFilesL3 = None
    if (sortedFilesL3 != None):
        numFilesL3 = len(sortedFilesL3)
        
    
    numNodesL1 = len(readContribs(os.path.join(contribsDirL1, sortedFilesL1[0]), wordBidict))
    numNodesL2 = None
    if (contribsDirL2 != None):
        numNodesL2 = len(readContribs(os.path.join(contribsDirL2, sortedFilesL2[0]), wordBidict))
    
    numNodesL3 = None
    if (contribsDirL3 != None):
        numNodesL3 = len(readContribs(os.path.join(contribsDirL3, sortedFilesL3[0]), wordBidict))
    
    # determine how many images have been writed by listing the image dir 
    # start on the subsequent image
    prevFiles = os.listdir(imageDir)
    start = len(prevFiles)
    
    # assume 2 layers for now
    nLayers = 1
    if (contribsDirL2 != None):
        nLayers += 1
    if (contribsDirL3 != None):
        nLayers += 1
        
    # digits in file name
    znum = 8
    
    c = start
    
    while (c < numFilesL1):
        contribFile = sortedFilesL1[c]
        path = os.path.join(contribsDirL1, contribFile)            
        contribs = readContribs(path, wordBidict)
        
        #            word cloud width              + padding width               + margins
        imageSize = (numNodesL1 * wordCloudSize[0] + (numNodesL1-1) * padding[0] + 2 * margins[0],
        #            word cloud height          + padding height           + margins
                     nLayers * wordCloudSize[1] + (nLayers-1) * padding[1] + 2 * margins[1])
        
        print ("Computing image for contrib at ", path)
        img = Image.new('RGB', imageSize, "white") # create new white image
        
        #start = time.time()
        
        # gen word cloud for each node
        for n in range(numNodesL1):
            
            wcImage = generateWordCloud(n, contribs, circle_mask, maxwords, normalize, normMin, normMax).to_image() #.resize(wordCloudSize) made mask 300x300 so don't need to resize
            
            # find desired location
            # 0,0 is at top left of image
            
            #           word cloud x start   + padding width  + left margin
            position = (n * wordCloudSize[0] + n * padding[0] + margins[0],
            #           word cloud y start TODO does not account for 2 layers
                        imageSize[1] - margins[1] - wordCloudSize[1])
            
            #start = time.time()
            img.paste(wcImage, position)
            #end = time.time()
            #print ("image paste = ", (end-start))
            
        # image number matches the iteration
        # make c 1-indexed not 0-indexed
        img.save(os.path.join(imageDir, str(c+1).zfill(znum) + '.png'), "PNG")
        
        #end = time.time() 
        #print ("word cloud gen = ", (end-start))
        
        c += 1
        
        
    if (finalImgL1 == None):
        # load last img
        path = os.path.join(imageDir, str(numFilesL1).zfill(znum) + '.png')
        finalImgL1 = Image.open(path)
        
    if (numFilesL2 != None):
        while (c < (numFilesL1 + numFilesL2) ):
        
            contribFile = sortedFilesL2[c - numFilesL1]
            path = os.path.join(contribsDirL2, contribFile)
            contribs = readContribs(path, wordBidict)
            
            # copy image from finalImgL1
            img = finalImgL1.copy()
                            
            print ("Computing image for contrib at ", path)
            
            nodesToSpace = (numNodesL1 - numNodesL2) / 2.0
            nodeOffset = round(nodesToSpace * wordCloudSize[0] + int(nodesToSpace) * padding[0])
            
            # gen word cloud for each node
            for n in range(numNodesL2):
                wcImage = generateWordCloud(n, contribs, circle_mask, maxwords, normalize, normMin, normMax).to_image() #.resize(wordCloudSize); made mask 300x300 so don't need to resize
                
                # find desired location
                # 0,0 is at top left of image
                
                #           offset to center L2 + word cloud x start     + padding width    + left margin
                position = (nodeOffset          + (n * wordCloudSize[0]) + (n * padding[0]) + margins[0],
                #           word cloud y start 
                            margins[1] + padding[1] + wordCloudSize[1])
            
                img.paste(wcImage, position)
            
            # image number matches the iteration
            # make c 1-indexed not 0-indexed
            img.save(os.path.join(imageDir, str(c+1).zfill(znum) + '.png'), "PNG")            
                
            c += 1
     
     
    if (numFilesL2 != None and finalImgL2 == None):
        # load last img
        path = os.path.join(imageDir, str(numFilesL1 + numFilesL2).zfill(znum) + '.png')
        finalImgL2 = Image.open(path)
        
    if (numFilesL3 != None):
        while (c < (numFilesL1 + numFilesL2 + numFilesL3) ):
    
            contribFile = sortedFilesL3[c - numFilesL1 - numFilesL2]
            path = os.path.join(contribsDirL3, contribFile)
            contribs = readContribs(path, wordBidict)
            
            # copy image from finalImgL1
            img = finalImgL2.copy()
                            
            print ("Computing image for contrib at ", path)
            
            nodesToSpace = (numNodesL1 - numNodesL3) / 2.0
            nodeOffset = round(nodesToSpace * wordCloudSize[0] + int(nodesToSpace) * padding[0])
            
            # gen word cloud for each node
            for n in range(numNodesL3):
                wcImage = generateWordCloud(n, contribs, circle_mask, maxwords, normalize, normMin, normMax).to_image() #.resize(wordCloudSize); made mask 300x300 so don't need to resize
                
                # find desired location
                # 0,0 is at top left of image
                
                #           offset to center L2 + word cloud x start     + padding width    + left margin
                position = (nodeOffset          + (n * wordCloudSize[0]) + (n * padding[0]) + margins[0],
                #           word cloud y start 
                            margins[1])
            
                img.paste(wcImage, position)
            
            # image number matches the iteration
            # make c 1-indexed not 0-indexed
            img.save(os.path.join(imageDir, str(c+1).zfill(znum) + '.png'), "PNG")            
                
            c += 1
            
            
