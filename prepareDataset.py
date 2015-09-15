# SciKit feature extraction
#   http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
# NLTK Tokenize
#   http://www.nltk.org/api/nltk.tokenize.html
# Python Argparse
#   https://docs.python.org/2/howto/argparse.html
# Loading data in Torch
#   http://fastml.com/loading-data-in-torch-is-a-mess/
#   See http://docs.h5py.org/en/latest/index.html for info on installing the h5py package


import os, sys, argparse, operator
import h5py
import numpy, scipy, sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import WhitespaceTokenizer


############################################
###            Data Reader               ###
############################################

class DataReader:
    """ 
    Read data from file or directory with options for documents in line or
    file format
    """
    
    DOC_FORMAT_LINE = "line"
    DOC_FORMAT_FILE = "file"

    def __init__(self, input, docsFormat, maxDocs, vocabFile, vocabSize):
        self.input = input
        self.docsFormat = docsFormat
        self.maxDocs = maxDocs
        self.vocabFile = vocabFile
        self.vocabSize = vocabSize
        self.corpus = []
        self.vocab = None
        
    def readData(self):
        """
        Check whether self.input is directory. If it is load all files within the
        directory. Otherwise just read self.input. This assumes they are pre processed.
        """
        # This must be done first because could return early due to doc limit
        if self.vocabFile != None: # read vocab
            self.readVocabFile()
                    
        print ("Reading data")
        filesToLoad = []
        if (os.path.isdir(self.input)):  
            for f in os.listdir(self.input):
                fpath = self.input+'/'+f
                if not os.path.isdir(fpath):
                    filesToLoad.append(fpath)
        else:
            filesToLoad.append(self.input)

        # moved this to another function because may need to parse vocab and can't have return statement
        self.loadCorpus(filesToLoad)
        
        # now go through the corpus and build a vocabulary of a given size based on word frequencies
        if self.vocab == None and self.vocabSize != None:
            self.buildVocab()
                
    
    def loadCorpus(self, filesToLoad):
        curNumDocs = 0
        for f in filesToLoad:
            with open(f, 'r') as datafile:
                text = datafile.read().strip()
                if (self.docsFormat == DataReader.DOC_FORMAT_LINE):
                    docs = text.splitlines()
                    # Somehow I was getting empty docs from the enwiki9cleaneddocs file
                    docs = [d for d in docs if d != ""]
                            
                    # If we have a docs limit and
                    # if the num docs we are trying to add will put us past the max
                    if (self.maxDocs != None and curNumDocs + len(docs) > self.maxDocs):
                        canAdd = self.maxDocs - curNumDocs
                        # add what we can
                        self.corpus.extend(docs[0:canAdd])
                        return
                    else:
                        self.corpus.extend(docs)
                        curNumDocs += len(docs)
                        
                elif (self.docsFormat == DataReader.DOC_FORMAT_FILE):
                    self.corpus.append(text)
                    
                    
    
    def buildVocab(self):
        self.vocabSize = int(self.vocabSize)
        print ("Building vocab from frequencies")
        # get tokenized corpus and get word counts
        self.tokenizedCorpus = []
        self.vocabSet = set()
        
        tokenizer = WhitespaceTokenizer()
        
        for doc in self.corpus:
            # tokenize doc
            docTokens = tokenizer.tokenize(doc)
            self.tokenizedCorpus.extend(docTokens)
        print ("  Tokenized corpus = ", len(self.tokenizedCorpus))

        # vocab for entire corpus
        self.fullVocab = set(self.tokenizedCorpus)
        print ("  Full vocab = ", len(self.fullVocab))
        
        self.vocabCounts = {}        
        # Extremely inefficient since has to iterate entire corpus for each word
        # generate counts for each word
        #for w in self.fullVocab:
        #    self.vocabCounts[w] = self.tokenizedCorpus.count(w)
        
        # for each word in corpus
        for w in self.tokenizedCorpus:
            if w in self.vocabCounts:
                self.vocabCounts[w] += 1
            else:
                self.vocabCounts[w] = 1


        # sort counts with most frequent first
        sortedCounts = sorted(self.vocabCounts.items(), key=operator.itemgetter(1), reverse=True)
        
        # generate vocab from first vocabSize words
        vocabCounts = sortedCounts[0:self.vocabSize]
        self.vocab = [e[0] for e in vocabCounts]
        print ("  vocab = ", self.vocab)
    
    
    def readVocabFile(self):
        print ("Reading vocab file")
        self.vocab = []
        with open(self.vocabFile, 'r') as vf:
            for line in vf:
                self.vocab.append(line.strip())
        
    
    def getCorpus(self):
        return self.corpus
    
        
    def getVocab(self):
        return self.vocab
        
        

############################################
###           Data Writer                ###
############################################

class DataWriter:
    
    DATA_OUT_FORMAT_HDF5 = "hdf5"
    DATA_OUT_FORMAT_TEXT = "text"

    def __init__(self, output, outputdir, outputformat):
        self.output = output
        self.outputdir = outputdir
        self.outputformat = outputformat
    
    
    def writeDataHDF5Dense(self, dataMatrix, datagroup, dimenData):
        """
        Here dataMatrix is a dense numpy array
        """
        chunkSizes = self.calculateChunks(dimenData)
        matrixdset = datagroup.create_dataset("matrix", data=dataMatrix, chunks=chunkSizes)
        chunkdset = datagroup.create_dataset("chunks", (len(chunkSizes),), dtype='i8')
        for i in range(len(chunkSizes)):
            chunkdset[i] = chunkSizes[i]
        
            
    def writeDataText(self, dataMatrix):
        """
        Write the data in text format
        """
        fname = self.getFilePath() + self.getFileExtension()
        with open(fname, 'w') as f:
            # write dimensions
            for i in range(dataMatrix.ndim):
                f.write(str(dataMatrix.shape[i]) + ' ')
            f.write('\n')
            
            for row in dataMatrix:
                for i in range(len(row)):
                    f.write(str(row[i]) + ' ')
                f.write('\n')
                
    
    def calculateChunks(self, dimenData, chunkSize=100000):
        """
        Return a tuple of chunks sizes in each dimension
        
        By default chunk sizes will be 100000, if the size of the
        matrix < 100000 in a dimension the chunksize will be set to
        the size in that dimension
        """
        chunks = []
        for d in dimenData:
            if (d > chunkSize):
                chunks.append(chunkSize)
            else:
                chunks.append(d)
        return tuple(chunks)        
                
                
    def writeWordDictText(self, wordDict):
        """
        write the word dict to a text file
        """
        wdictfname = self.getFilePath('worddict') + ".txt"
        with open(wdictfname, 'w') as f:
            for k in wordDict.keys():
                f.write(str(k) + ' ' + str(wordDict[k]) + '\n')
                
        
    def getFilePath(self, fileIdentifier=None):
        """
        Get the file path based on the output prefix, output dir
        The fileIdentifier if given will replace the generic output file
        """
        filePath = None
        if (self.outputdir != None):
            if not os.path.exists(self.outputdir):
                os.mkdir(self.outputdir)
                
            if (fileIdentifier != None):
                filePath = self.outputdir + '/' + fileIdentifier
            else:
                filePath = self.outputdir + '/' + self.output
                
        else:
            if (fileIdentifier != None):
                filePath = fileIdentifier
            else:
                filePath = self.output
            
        return filePath
    
    
    def getFileExtension(self):
        """
        Get the file extension from the output format
        """
        if (self.outputformat == DataWriter.DATA_OUT_FORMAT_HDF5):
            return '.h5'
        elif (self.outputformat == DataWriter.DATA_OUT_FORMAT_TEXT):
            return '.txt'
            
    

class OneOfNDataWriter(DataWriter):
    """
    Handle writing 1 of N data to file in both hdf5 and text formats
    in both sparse and dense representations.
    
    Writing to a text file will always be dense.
    """
    
    def __init__(self, output, outputdir, outputformat):
        super(self.__class__, self).__init__(output, outputdir, outputformat)
    
    
    def writeData(self, dataMatrix, wordDict, sparse):
        """
        Write the data to a file in one of the given formats
        dims = None by default, but can be overridden with tuple
        """
        if (self.outputformat == DataWriter.DATA_OUT_FORMAT_HDF5):
            # write to hdf5 file
            print ("Writing data as hdf5")
            self.writeDataHDF5(dataMatrix, wordDict, sparse)
                      
        elif (self.outputformat == DataWriter.DATA_OUT_FORMAT_TEXT):
            # write to text file
            print ("Writing data as text")
            self.writeDataText(dataMatrix)
        
        self.writeWordDictText(wordDict)
        
    
    def writeDataHDF5(self, dataMatrix, wordDict, sparse):
        """
        Write the data in HDF5 format
        
        /meta/type = [1on=0|bow=1]
        /meta/dim = list(dim0, dim1, dim2, ... , dimn)
        /meta/sparse = [0 = dense | 1 = sparse]
        /data/matrix
  
        By default the data will be chunked in (100, 100, ... 100) pieces
        
        Encodings
        1 of N: 
                Dense - The data matrix will be size (m x n) where m is the number of words in the corpus and n is the size of the vocabulary.
                Sparse - ?
                       
        """
        fname = self.getFilePath() + self.getFileExtension()
        f = h5py.File(fname, 'w')
        sklearn.preprocessing.normalize
        metagroup = f.create_group("meta")
        
        # set the type
        ptype = 0 #DataProcessor.PROCESSOR_TYPE_1ON
        typedset = metagroup.create_dataset("type", (1,), dtype='i8')
        typedset[0] = ptype
        
        # sparse?
        sparsedset = metagroup.create_dataset("sparse", (1,), dtype='i8')
        if (sparse):
            sparsedset[0] = 1
        else:
            sparsedset[0] = 0
        
        # dimens
        dimenData = numpy.empty((dataMatrix.ndim,))
        for i in range(dataMatrix.ndim):
            dimenData[i] = dataMatrix.shape[i]
        dimendset = metagroup.create_dataset("dim", data=dimenData)
            
        datagroup = f.create_group("data")
        if (sparse):
            # data is in sparse format
            self.writeDataHDF5Sparse(dataMatrix, datagroup)
        else:
            # data is in dense format
            self.writeDataHDF5Dense(dataMatrix, datagroup, dimenData)
            
        print ("Wrote hdf5 to", fname)
        
    
    def writeDataHDF5Sparse(self, dataMatrix, datagroup):
        """
        Here dataMatrix is a scipy dok sparse matrix. 
        Dimens are (patterns, elements in pattern)
        """
        
        # dataMatrix.nonzero() returns a pair [array rows][array cols] which are nonzeros
        # I select the cols, which correspond to the index of the word
        # these appear to be ordered left-to-right, top-to-bottom
        # http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.dok_matrix.nonzero.html#scipy.sparse.dok_matrix.nonzero
        datadset = datagroup.create_dataset("matrix", data=dataMatrix.nonzero()[1])        
        
        
        
class BagOfWordsDataWriter(DataWriter):
    """
    Responsible for writing data to HDF5 and text formats
    in both sparse or dense representations
    
    Writing to a text file will always be dense.
    """
    
    def __init__(self, output, outputdir, outputformat):
        super(self.__class__, self).__init__(output, outputdir, outputformat)
        

    def writeData(self, dataMatrix, wordDict, sparse):
        """
        Write the data to a file in one of the given formats
        dims = None by default, but can be overridden with tuple
        """
        
        if (self.outputformat == DataWriter.DATA_OUT_FORMAT_HDF5):
            # write to hdf5 file
            print ("Writing data as hdf5")
            self.writeDataHDF5(dataMatrix, wordDict, sparse)
                      
        elif (self.outputformat == DataWriter.DATA_OUT_FORMAT_TEXT):
            # write to text file
            print ("Writing data as text")
            self.writeDataText(dataMatrix)
        
        self.writeWordDictText(wordDict)
        
        
    def writeDataHDF5(self, dataMatrix, wordDict, sparse):
        """
        Write the data in HDF5 format
        
        /meta/type = [1on=0|bow=1]
        /meta/dim = list(dim0, dim1, dim2, ... , dimn)
        /meta/sparse = [0 = dense | 1 = sparse]
        /data/matrix
  
        By default the data will be chunked in (100, 100, ... 100) pieces
        
        Encodings
            Bag of Words: The data matrix will be size (d x n) where d is the number of documents and n is the size of the vocabulary
        """
        fname = self.getFilePath() + self.getFileExtension()
        f = h5py.File(fname, 'w')
        
        metagroup = f.create_group("meta")
        
        # set the type
        ptype = 1 #DataProcessor.PROCESSOR_TYPE_BOW
        typedset = metagroup.create_dataset("type", (1,), dtype='i8')
        typedset[0] = ptype
        
        # dimens
        dimenData = numpy.empty((dataMatrix.ndim,))
        for i in range(dataMatrix.ndim):
            dimenData[i] = dataMatrix.shape[i]
        dimendset = metagroup.create_dataset("dim", data=dimenData)
        
        # sparse?
        sparsedset = metagroup.create_dataset("sparse", (1,), dtype='i8')
        if (sparse):
            sparsedset[0] = 1
        else:
            sparsedset[0] = 0
        
        # data
        datagroup = f.create_group("data")
        if (sparse):
            # data is in sparse format
            self.writeDataHDF5Sparse(dataMatrix, datagroup)
        else:
            # data is in dense format
            self.writeDataHDF5Dense(dataMatrix, datagroup, dimenData)
            
        print ("Wrote hdf5 to", fname)
    
    
    def writeDataHDF5Sparse(self, dataMatrix, datagroup):
        """
        Here dataMatrix is a scipy csr sparse matrix. 
        Dimens are (patterns, elements in pattern)
        
        write:
        m.data
        m.indices
        m.indptr
        """
        datadset = datagroup.create_dataset("data", data=dataMatrix.data)
        indicesDset = datagroup.create_dataset("indices", data=dataMatrix.indices)
        indptrDset = datagroup.create_dataset("indptr", data=dataMatrix.indptr)
        
    
    def writeDataHDF5Dense(self, dataMatrix, datagroup, dimenData):
        """
        Here dataMatrix is a dense numpy array
        """
        chunkSizes = self.calculateChunks(dimenData)
        matrixdset = datagroup.create_dataset("matrix", data=dataMatrix, chunks=chunkSizes)
        chunkdset = datagroup.create_dataset("chunks", (len(chunkSizes),), dtype='i8')
        for i in range(len(chunkSizes)):
            chunkdset[i] = chunkSizes[i]
            
        
    def writeDataText(self, dataMatrix):
        """
        Write the data in text format
        """
        fname = self.getFilePath() + self.getFileExtension()
        with open(fname, 'w') as f:
            # write dimensions
            # probably 2d but just in case
            for i in range(dataMatrix.ndim):
                f.write(str(dataMatrix.shape[i]) + ' ')
            f.write('\n')
            
            for row in dataMatrix:
                #f.write('[')
                for i in range(len(row)):
                    f.write(str(row[i]) + ' ')
                    #if (i != len(row)-1):
                    #    f.write(',')
                f.write('\n')



############################################
###          Data Processors             ### 
############################################

class DataProcessor:

    PROCESSOR_TYPE_1ON = "1on"
    PROCESSOR_TYPE_BOW = "bow"
    PROCESSOR_TYPE_W2V = "w2v"

    def __init__(self, processorType):
        self.processorType = processorType
        self.corpus = []
        self.vocab = None
        self.wordDict = None
        self.sparse = None # boolean will be set in process
        self.dims = None # will be set if doing sparse calc
        self.data = None # will be numpy array

        # will read / write data
        self.reader = None
        self.writer = None
        
    
    def readData(self, input, docsFormat, maxDocs, vocabFile, vocabSize):
        """
        Load the data using a data reader
        If maxDocs is set to None all the docs will be loaded
        """
        self.reader = DataReader(input, docsFormat, maxDocs, vocabFile, vocabSize)
        self.reader.readData();
        self.corpus = self.reader.getCorpus()
        self.vocab = self.reader.getVocab()
        #print ("Num docs =", len(self.corpus))
        #print ("Vocab = ", self.vocab)
        
        
    def writeData(self, output, outputdir, outputformat):
        """
        Write the data using a data writer
        """
        if (type(self) == OneOfNProcessor):
            self.writer = OneOfNDataWriter(output, outputdir, outputformat)
        elif (type(self) == BagOfWordsProcessor):
            self.writer = BagOfWordsDataWriter(output, outputdir, outputformat)
           
        self.writer.writeData(self.getDataMatrix(), self.wordDict, self.sparse)
        
    
    def buildWordDict(self):
        """ 
        Return map of word to index
        """
        print ("  Building worddict")
        vocabList = list(self.vocab)
        self.wordDict = {}
        for i in range(len(vocabList)):
            self.wordDict[vocabList[i]] = i
        return self.wordDict


    def normalizeData(self):
        """
        Normalizes the data in self.data
        """
        print ("Normalizing data")
        #sklearn.preprocessing.normalize(self.data, 'l2', 0, False)
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        min_max_scaler.fit_transform(self.data, False)
                

    def getProcessorType(self):
        return self.processorType



class OneOfNProcessor(DataProcessor):
    # TODO: update this so it respects the loaded --vocab
    """
    Processor to create input data where each word is designated by a 1 of N encoding
    Given a vocabulary of size m, the nth word will be represented by an array of m elements
    where each of the elements are 0 except for the nth element, which will be a 1
    """
    def __init__(self):
        super(self.__class__, self).__init__(DataProcessor.PROCESSOR_TYPE_1ON)
        self.numWords = None # placehold to make explicit that var exists
        self.vocabSize = None
        

    def process(self, sparse=True, normalize=False):
        """
        For document in corpus, build a vocabulary, and generate sklearn.preprocessing.normalize
        vectors for each word in order. Each input document will have a corresponding 
        output document.
        """
        print ("Processing data for 1 of N")

        corpusTokens = self.tokenizeCorpus()
        self.buildVocabulary(corpusTokens)
        self.buildWordDict()

        # Set the number of words based on the tokens
        self.numWords = len(corpusTokens)
        self.vocabSize = len(self.vocab)
        
        print ("  Vocab size =", self.vocabSize)
        print ("  Corpus size =", self.numWords)
        
        self.sparse = sparse
        if (sparse):
            self.buildSparseArray(corpusTokens)
        else:
            self.buildDenseArray(corpusTokens)
            
        if (normalize):
            self.normalizeData()
    
    
    def buildSparseArray(self, corpusTokens):
        print ("  Building sparse array")
        self.data = scipy.sparse.dok_matrix((self.numWords, self.vocabSize))
        print ("  self.data=", type(self.data), self.data.shape, self.data.ndim)
        for i in range(len(corpusTokens)):
            token = corpusTokens[i]
            self.data[i,self.wordDict[token]] = 1
            
    
    def buildDenseArray(self, corpusTokens):
        print ("  Building dense array")
        self.data = numpy.zeros((self.numWords, self.vocabSize))
        for i in range(len(corpusTokens)):
            token = corpusTokens[i]
            self.data[i][self.wordDict[token]] = 1 
        
            
    def tokenizeCorpus(self):
        """
        Returns a 1D list of tokens in the entire corpus.
        Each doc is enclosed in tags (<doc> </doc>)
        """
        print ("  Tokenizing corpus")
        corpusTokens = []
        
        print (" ", len(self.corpus), "docs")
        for dnum in range(len(self.corpus)):
            doc = self.corpus[dnum]
            if (dnum % 1000 == 0):
                print ("  Tokenizing doc:", dnum)
            # Begin doc
            corpusTokens.append("<doc>")
            corpusTokens.extend(self.tokenizeDoc(doc))
            # end doc
            corpusTokens.append("</doc>")
        
        return corpusTokens
        
        
    def tokenizeDoc(self, doc):
        """
        Get the tokens (words) from the doc
        uses nltk.
        """
        #print ("Tokenizing doc")
        tokenizer = WhitespaceTokenizer()
        docTokens = tokenizer.tokenize(doc)
        return docTokens
        
        
    def buildVocabulary(self, corpusTokens):
        """
        Build a vocabulary of all unique words in the corpus
        """
        print ("  Building vocabulary for corpus")
        self.vocab = set()
        for token in corpusTokens:
            self.vocab.add(token)
        return self.vocab
    
    
    def getCorpusSize(self):
        """ 
        Returns the number of words in the corpus
        """
        return self.numWords
        
        
    def getDataMatrix(self):
        """
        Returns concatenated documents with begin
        and end tags
        """
        return self.data
        

            
        
class BagOfWordsProcessor(DataProcessor):
    """
    Creates an output dataset where each document is represented by a vector with the 
    counts of words in that document. This assumes multiple documents are given as input
    """
    def __init__(self):
        super(self.__class__, self).__init__(DataProcessor.PROCESSOR_TYPE_BOW)
        self.X = None #placeholder just to make is clear that these vars exist
        self.tfidf = None
       

    def process(self, sparse=False, normalize=False):
        """
        need loaded data in form of 
        corpus [
            'doc1',
            'doc2',
            'doc3',
            ...
            'docn'
        ]
        
        """
        
        print ("Processing data for bag of words")
        self.sparse = sparse
        
        print ("  Calculating word counts")
        vectorizer = CountVectorizer(min_df=0, vocabulary=self.vocab)
        
        #print (self.corpus)
        
        self.X = vectorizer.fit_transform(self.corpus)
        #print (self.X.toarray())
        
        # vocab could be set from file
        if self.vocab == None:
            self.vocab = vectorizer.get_feature_names()
        
        self.wordDict = vectorizer.vocabulary_
        #print (self.wordDict)
        
        # do tfidf on the counts 
        print ("  Calculating tfidf")
        transformer = TfidfTransformer()
        self.tfidf = transformer.fit_transform(self.X)
        
        #print (self.tfidf.toarray())
        
        if (sparse):
            self.data = self.tfidf
        else:
            self.data = self.tfidf.toarray()
        
        if (normalize):
            self.normalizeData()
        
        #print (self.data)
        
        
    def getDataMatrix(self):
        """
        Get the computed data that will be written to disk
        """
        return self.data
            

        
class Word2VecProcessor(DataProcessor):
    """
    Calls the Word2Vec code to create a vector representation for each word
    """
    def __init__(self):
        super(self.__class__, self).__init__(DataProcessor.PROCESSOR_TYPE_W2V)
        
        

def getProcessor(procType):
    procType = procType.lower()
    processor = None
    if (procType == DataProcessor.PROCESSOR_TYPE_BOW):
        processor = BagOfWordsProcessor()
    elif (procType == DataProcessor.PROCESSOR_TYPE_1ON):
        processor = OneOfNProcessor()
    elif (procType == DataProcessor.PROCESSOR_TYPE_W2V):
        processor = Word2VecProcessor()
    
    return processor
        
        
        
if __name__ == "__main__":
    """
    input: file name or directory containing files to be processed
    Output: a file with processed data ready to be used as input for other applications

    Processing types: bag of words[bow], 1 of N [1on], Word2Vec [w2v]
    """

    parser = argparse.ArgumentParser(prog='PrepareDataset')
    
    parser.add_argument('--input', help='file name or directory containing files to be processed')
    parser.add_argument('--output', help='name of file prefix to write to (default=out)', default='out')
    parser.add_argument('--outputdir', help='name of file dir to write to. preferred for 1 of N. (default=outdata)', default='outdata')
    parser.add_argument('--processor', help='how should the data be processed [bow, 1on, w2v] (default=None)', default=None)
    parser.add_argument('--docsformat', help='what represents a single document [line, file] (default=line)', default=DataReader.DOC_FORMAT_LINE)
    parser.add_argument('--outputformat', help='what format the data output should be [hdf5, text] (default=hdf5)', default=DataWriter.DATA_OUT_FORMAT_HDF5)
    parser.add_argument('--maxdocs', help='the maximum number of documents to load, default is all', default=None)
    
    # corpus options
    parser.add_argument('--vocab', help='file to load corpus from')
    parser.add_argument('--vocabsize', help='give the maximum number of words permissible in the vocab')
    #parser.add_argument('--vocabmaxfreq', help='', default=None)
    #parser.add_argument('--vocabminfreq', help='', default=None)
    
    # sparse or dense
    parser.add_argument('--sparse', dest='dosparse', action='store_true', help='process the data in sparse format')
    parser.add_argument('--dense', dest='dosparse', action='store_false', help='process the data in dense format')
    parser.set_defaults(dosparse=True)
    
    # normalize or not
    parser.add_argument('--normalize', dest='donormalize', action='store_true', help='normalize the data after processing')
    parser.set_defaults(donormalize=False)

    args = parser.parse_args()
    
    
    # input
    if (args.input == None):
        print ("Must supply input file or directory")
        sys.exit(1)
    else:
        print ("Using input =", args.input)
    
    # output    
    print ("Using output =", args.output)
        
    # outputdir
    print ("Using outdata =", args.outputdir)
        
    # processor
    if (args.processor == None):
        print ("Must set processor type")
        sys.exit(1)   
    else:
        print ("Using processor =", args.processor)
        
    # docsformat
    print ("Using docsformat =", args.docsformat)
        
    # outputformat
    print ("Using outputformat =", args.outputformat)
            
    # maxdocs
    maxDocs = None
    if (args.maxdocs == None):
        print ("Loading all docs")
    else:
        print ("Loading maximum", args.maxdocs, "docs")
        maxDocs = int(args.maxdocs)
    
    # try to initialize a processor        
    processor = getProcessor(args.processor)
    if (processor == None):
        print ("Incorrect processor type")
        sys.exit(1)
    
    processor.readData(args.input, args.docsformat, maxDocs, args.vocab, args.vocabsize)
    processor.process(args.dosparse, args.donormalize)
    processor.writeData(args.output, args.outputdir, args.outputformat);
    
    
