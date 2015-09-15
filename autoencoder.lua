-- 
-- Autoencoder
-- 
-- 
-- 

--require 'hdf5'
require 'nn'
require 'optim'
--require 'cunn'

require 'input'
require 'logger'
require 'data_loader'
require 'data_writer'
require 'model_builder'
require 'utils'


-- 
-- find the vector representation of a given word. 
-- if the word to vec mapping is supplied, it is first priority
-- then a sparse or dense vector from the wordDict is created
--
function getVectorRepresentation(word, dim, word2vec, wordDict)
    local pat = nil
    if (word2vec ~= nil) then
        pat = word2vec[word]
    else
        -- generate representation
        if (DataLoader.isSparse) then
          -- Get a sparse representation of the word encoding
            pat = Utils:getSparseVector(val, 1)
        else
            pat = torch.Tensor(dim) 
    	    pat[wordDict[word]] = 1
	    end
    end
    return pat
end


-- 
-- pushes the given vector through the network and returns the hidden layer activation
-- assumes the network consist of only one hidden layer in addition to any dropout
-- or transfer functions added
--
function getHiddenActivation(model, vec)
    local layerMap = ModelBuilder:getLayerMap()
    
    --local vecs = {}
    -- and feed it through each layer
    local lastOut = vec
    for i = 1, table.getn(layerMap) - 1 do
        local layerToPushThrough = layerMap[i]
        -- Push through layer (including transfer but NOT dropout) and save output
        for j = 1, layerToPushThrough do
            if (torch.typename(model:get(j)) ~= 'nn.Dropout') then
                lastOut = model:get(j):forward(lastOut)
            end
        end
    end
    
    return lastOut:clone()
end


--
-- This function tries to determine what the hidden nodes have 
-- learned in relation to the inputs
--
function determineContributions(model, contribPath, model_l1, model_l2)
    Logger:info("Determining hidden node contributions and writing to " .. contribPath)
    
    DataWriter:openContribsFile(contribPath)
    
    local layerMap = ModelBuilder:getLayerMap()
    local layers = {}
    -- find the deepest layer necessary from the layer map
    -- load all of those layers into the `layers` array
    local deepestRequired = layerMap[table.getn(layerMap)-1]
    for i = 1, deepestRequired do
        layers[i] = model:get(i)
    end    
    
    -- The cols correspond to input nodes
    -- Rows correspond to hidden nodes
    local modelWeights = layers[1].weight
    local inputDimenM3 = modelWeights:size(2)
    local hiddenDimenM3 = modelWeights:size(1)
        
    -- for each hidden node
    for h = 1, hiddenDimenM3 do
    
        -- Denominator will be same for this hidden node
        local sum = 0
        for j = 1, inputDimenM3 do
            sum = sum + math.pow(modelWeights[h][j], 2)
        end
        local denom = math.sqrt(sum)
        
        local hiddenContrib = torch.Tensor(inputDimenM3)
        for i = 1, inputDimenM3 do
            local numer = modelWeights[h][i]
            hiddenContrib[i] = numer / denom
        end
       

        if (model_l2 ~= nil) then
            --print ("This is a third layer")
                        
            -- this is a 3nd layer
            -- compute first layer by 
            -- Weights * input = activations
            
            -- weights is 15 x 280
            local modelL2Weights = model_l2:get(1).weight
            --print (weights)
            
            -- b is 15 x 1
            -- = (inputDimen x 1)
            -- print ("in dimen = ", inputDimen)
            local b = hiddenContrib:resize(inputDimenM3, 1)
            --print (b)
            
            -- 275 x 1
            -- = (# words x 1)
            local prevLayer = torch.gels(b, modelL2Weights)
            
            local model_l2InputDimen = modelL2Weights:size(2)
            --print ("model l1 inputDimen = ", model_l1InputDimen)
            prevLayer:resize(model_l2InputDimen)
            
            -- TODO: now prev layer is the max activation for l2            
            
            if (model_l1 == nil) then
                print ("Model l1 cannot be nil!")
                return
            end
            
            local inputDimenM2 = modelL2Weights:size(2)
            local hiddenDimenM2 = modelL2Weights:size(1)
            --print ("computing for model_l1")
            -- this is a 2nd layer
            -- compute first layer by 
            -- Weights * input = activations
            
            -- weights is 15 x 280
            local modelL1Weights = model_l1:get(1).weight
            --print (weights)
            
            -- b is 15 x 1
            -- = (inputDimen x 1)
            -- print ("in dimen = ", inputDimen)
            local b = prevLayer:resize(inputDimenM2, 1)
            --print (b)
            
            -- 275 x 1
            -- = (# words x 1)
            local input = torch.gels(b, modelL1Weights)
            
            local model_l1InputDimen = modelL1Weights:size(2)
            --print ("model l1 inputDimen = ", model_l1InputDimen)
            input:resize(model_l1InputDimen)
            
            
            DataWriter:writeContrib(h, input)
            
                 
        -- this is the contribution of each input to hidden node h
        elseif (model_l1 ~= nil) then
        
            --print ("computing for model_l1")
            -- this is a 2nd layer
            -- compute first layer by 
            -- Weights * input = activations
            
            -- weights is 15 x 280
            weights_l1 = model_l1:get(1).weight
            --print (weights)
            
            -- b is 15 x 1
            -- = (inputDimen x 1)
            -- print ("in dimen = ", inputDimen)
            local b = hiddenContrib:resize(inputDimen, 1)
            --print (b)
            
            -- 275 x 1
            -- = (# words x 1)
            local input = torch.gels(b, weights_l1)
            
            local model_l1InputDimen = weights_l1:size(2)
            --print ("model l1 inputDimen = ", model_l1InputDimen)
            input:resize(model_l1InputDimen)
            
            DataWriter:writeContrib(h, input)
            --print (input)            
        else 
            -- no need to calculate previous layer 
            DataWriter:writeContrib(h, hiddenContrib)
        end
        
    end
    
    DataWriter:closeContribsFile()
end


-- 
-- Extract the hidden layer data for each word, uses all words in the vocabulary
-- or the word2vec map if it is provided
--
function extractModelData(model, wordDict, word2vec, featuresPath)
    Logger:info("Extracting data from network and writing to " .. featuresPath)
    
    -- print (model)
    local layerMap = ModelBuilder:getLayerMap()
    local layers = {}
    -- find the deepest layer necessary from the layer map
    -- load all of those layers into the `layers` array
    local deepestRequired = layerMap[table.getn(layerMap)-1]
    for i = 1, deepestRequired do
        layers[i] = model:get(i)
    end    
    
    -- Determine the dimension of the output
    local nPats = 0
    for _ in pairs(wordDict) do nPats = nPats + 1 end
    -- TODO: generalize this as an array holding dimens of many layers
    local inputDimen = layers[1].weight:size(2)
    local hiddenDimen = layers[1].weight:size(1)
    
    --                                   # patterns, # features / pat
    DataWriter:openFeaturesFile(featuresPath, nPats, hiddenDimen)
    
    local count = 1
    -- now for each word in the dataset
    for key, val in pairs(wordDict) do
        local pat = getVectorRepresentation(key, inputDimen, word2vec, wordDict)
        local vecs = {}
        
        -- and feed it through each layer
        local lastLayer = 0
        local lastOut = pat
        for i = 1, table.getn(layerMap) - 1 do
            local layerToPushThrough = layerMap[i]
            -- Push through layer (including transfer but NOT dropout) and save output
            for j = lastLayer + 1, layerToPushThrough do
                if (torch.typename(layers[j]) ~= 'nn.Dropout') then
                    lastOut = layers[j]:forward(lastOut)
                end
            end
            vecs[#vecs+1] = lastOut
            lastLayer = layerToPushThrough
        end
        
        DataWriter:writeFeatures(key, vecs)

        if (count % 1000 == 0) then
            print ("Processed pattern", count)
        end
        count = count + 1
    end
    
    DataWriter:closeFeaturesFile()    
end

--
-- Get the activations at the given layer of every input in the dataset
-- This is useful in generating input for stacked autoencoders
--
function getLayerActivations(model, layerInd, dataset, activationsPath)
    -- for each pattern in dataset
    -- push through to layerInd layer
    -- save output to form dataset for next autoencoder layer
    
    print ("Writing layer activations of layer " .. layerInd .. " to " .. activationsPath)
    Logger:info("Writing layer activations of layer " .. layerInd .. " to " .. activationsPath)

    local layerMap = ModelBuilder:getLayerMap()
    local layers = {}
    for i = 1, layerMap[layerInd] do
        layers[i] = model:get(i)
    end    
    
    local hiddenDimen = layers[1].weight:size(1)
    DataWriter:openActivationsFile(activationsPath, dataset:size(), hiddenDimen)
    
    for d = 1, dataset:size() do
        local lastLayer = 0
        local lastOut = dataset[d][1]
        for l = 1, layerInd do
            local layerToPushThrough = layerMap[l]
            -- Push through layer (including transfer but NOT dropout) and save output
            for j = lastLayer + 1, layerToPushThrough do
                if (torch.typename(layers[j]) ~= 'nn.Dropout') then
                    lastOut = layers[j]:forward(lastOut)
                end
            end
            lastLayer = layerToPushThrough
        end
        
        DataWriter:writeActivation(lastOut)
    end
    
    DataWriter:closeActivationsFile()
end


--
--    c1 c2 c3 c4 c5 -- intra cluster distances 
-- c1 o  x  x  x  x
-- c2    o  x  x  x
-- c3       o  x  x
-- c4          o  x
-- c5             o
--
function checkCentroids(model, clusters, currentItrs, word2vec, wordDict)
    -- build structure indexed by cluster and then vector in cluster
    print ("Checking centroids", currentItrs)
    
    local clusterActivationVecs = {}
    for c = 1, table.getn(clusters) do
        local activationVecs = {}
        local cluster = clusters[c]
        -- for each word in cluster, generate vector
        for w = 1, table.getn(cluster) do
            local activationVec = getHiddenActivation(
                                model,
                                getVectorRepresentation(cluster[w], model:get(1).weight:size(2), word2vec, wordDict))
            activationVecs[#activationVecs + 1] = activationVec
        end
        clusterActivationVecs[#clusterActivationVecs + 1] = activationVecs
    end

    -- calculate mean of cluster
    local clusterMeans = {}
    -- for each cluster get the vectors associated with it
    for c = 1, table.getn(clusterActivationVecs) do    
        --print ("c = ", c)
        local clusterVecs = clusterActivationVecs[c]
        -- TODO ensure this clusterMean is initialized to 0s
        local clusterMean = torch.Tensor(clusterVecs[1]:size(1)):zero()
        local numVecsInCluster = table.getn(clusterVecs)

        -- for each vector in cluster
        for v = 1, numVecsInCluster do
            clusterMean:add(clusterVecs[v])
            --print ("clusterVecs[v] with v = ", v, ", vecs = ", clusterVecs[v])
            --print ("clusterMean[v] with v = ", v, ", mean = ", clusterMean)
        end
        
        clusterMean:div(numVecsInCluster)
        --print ("clusterMean post divide", clusterMean)
        clusterMeans[#clusterMeans + 1] = clusterMean
    end
    
    -- calculate sum of squared error intra cluster
    local intraClusterSSE = {}
    for c = 1, table.getn(clusterActivationVecs) do
        -- compute SSE 
        local SSE = 0
        local vecs = clusterActivationVecs[c]
        for v = 1, table.getn(vecs) do
            -- find difference from computed activation and cluster mean
            local err = clusterMeans[c] - vecs[v]
            -- square the error
            err:pow(2)
            
            --print ("clusterMeans[c] = ", clusterMeans[c], ", err = ", err)
            -- add the sum of the square for each element to the SSE total
            SSE = SSE + err:sum()
        end
        -- divide SSE by number of vecs to get error per vector
        intraClusterSSE[c] = SSE / table.getn(vecs)
        
        --print ("IntraClusterSSE for cluster " .. c .. " = " .. intraClusterSSE[c] .. ", pre norm SSE = " .. SSE .. ", num vecs = " .. table.getn(vecs))
        Logger:info("IntraClusterSSE for cluster " .. c .. " = " .. intraClusterSSE[c])
    end
    
    local avgIntraClusterSSE = 0
    for i = 1, table.getn(intraClusterSSE) do
        avgIntraClusterSSE = avgIntraClusterSSE + intraClusterSSE[i]
    end
    avgIntraClusterSSE = avgIntraClusterSSE / table.getn(intraClusterSSE)
    Logger:info("AvgIntraClusterSSE = " .. avgIntraClusterSSE)
    --print ("AvgIntraClusterSSE = ", avgIntraClusterSSE)
    
    -- compute inter cluster distances
    -- compute table and don't need to compute pair if row index >= col index
    local interClusterDistances = {}
    for c = 1, table.getn(clusterMeans) do
        interClusterDistances[c] = {}
    end
    
    -- (5! / 2!*(5-2)! = 120 / (2*6) = 120 / 12 = 10   
    local avgInterClusterDistance = 0
    local pairs = 0 -- TODO can be computed with n choose k but don't know how to do factorial with torch
    for a = 1, table.getn(clusterMeans) do
        for b = a + 1, table.getn(clusterMeans) do
            local dist = torch.dist(clusterMeans[a], clusterMeans[b])
            avgInterClusterDistance = avgInterClusterDistance + dist
            interClusterDistances[a][b] = dist
            pairs = pairs + 1
        end
    end
    
    avgInterClusterDistance = avgInterClusterDistance / pairs
    
    -- a,b:dist;a,c:dist;
    -- TODO: write to file to compute trends from?
    --print ("AvgInterClusterDistance = " .. avgInterClusterDistance)
    Logger:info("AvgInterClusterDistance = " .. avgInterClusterDistance)
end


--
-- save necessary data at this stage
--
function saveOperation(model, model_l1, model_l2, itrs, params, modelParams, wordDict, word2vec, dataset)
    local outFile = modelParams.data.model .. tostring(itrs) .. "itrs.txt"
    Logger:info("Saving model to file " .. outFile)
    torch.save(outFile, model)
    
    local contribsSaveItrs = tonumber(modelParams.logging.contribsSaveItrs)
    local featuresSaveItrs = tonumber(modelParams.logging.featuresSaveItrs)
    local hiddenActivationsSaveItrs = tonumber(modelParams.logging.hiddenActivationsSaveItrs)
   
    if (contribsSaveItrs ~= nil and itrs % contribsSaveItrs == 0 and params.contribs == 'yes') then
        local contribsFile = modelParams.data.contribs .. tostring(itrs) .. "itrs.txt"
        determineContributions(model, contribsFile, model_l1, model_l2)
    end
    
    if (featuresSaveItrs ~= nil and itrs % featuresSaveItrs == 0 and params.extract == 'yes') then
        local featuresFile = modelParams.data.features .. tostring(itrs) .. "itrs.txt"
        extractModelData(model, wordDict, word2vec, featuresFile)
    end
    
    if (hiddenActivationSaveItrs ~= nil and itrs % hiddenActivationsSaveItrs == 0 and params.activations == 'yes') then
        local activationsFile = modelParams.data.activations .. tostring(itrs) .. "itrs.txt"
        getLayerActivations(model, 1, dataset, activationsFile)
    end
end


--
-- Train autoencoder for the given number of iterations on the model provided
-- writing the current model every 'saveItrs' iterations
-- if the clusters variable is not nil then they will be computed every 'saveItrs' as well
-- 
-- TODO: update this to do the correct number of iterations so can save on arbitrary itrs
--
--
function trainAutoencoder(model, model_l1, model_l2, trainer, itrsCompleted, params, modelParams, dataset, clusters, word2vec, wordDict)

    local itrs = tonumber(modelParams.training.itrs) 
    
    local modelSaveItrs = tonumber(modelParams.logging.modelSaveItrs)
    
    -- do itrs iterations saving periodically
    local i = itrsCompleted
    if (i == itrs) then
        print ("Already at " .. i .. " iterations")
    end
    
    local firstSet = modelSaveItrs
    -- if doing this pushes us past the max, just add as many as necessary
    if (itrs < itrsCompleted + firstSet) then
        firstSet = itrs - itrsCompleted
    end
    for j = i + 1, firstSet + itrsCompleted do
        trainer.maxIteration = 1
        trainer:train(dataset)
        print ("Finished iteration " .. j .. " with error " .. trainer:getError())
        Logger:info("Finished iteration " .. j .. " with error = " .. trainer:getError())        
    end
    i = i + firstSet
    
    saveOperation(model, model_l1, model_l2, i, params, modelParams, wordDict, word2vec, dataset)
    
    --if (clusters ~= nil) then
    --    checkCentroids(model, clusters, i, word2vec, wordDict)
    --end
        
    while i < itrs do
        local itrsTodo = modelSaveItrs
        if i + itrsTodo > itrs then
            itrsTodo = itrs - i
        end
        
        trainer.maxIteration = itrsTodo
        trainer:train(dataset)
        
        i = i + itrsTodo
        
        print ("Finished iteration " .. i .. " with error " .. trainer:getError())
        Logger:info("Finished iteration " .. i .. " with error = " .. trainer:getError())

        saveOperation(model, model_l1, model_l2, i, params, modelParams, wordDict, word2vec, dataset)
    
        --if (clusters ~= nil) then
        --    checkCentroids(model, clusters, i, word2vec, wordDict)
        --end
        
    end
end



-- recast every sample in the dataset as a cuda tensor
function convertDatasetForCuda(dataset) 
    for i = 1, dataset:size() do
        dataset[i][1] = dataset[i][1]:cuda()
        dataset[i][2] = dataset[i][2]:cuda()
    end
end


-- recast every sample in the dataset as a double tensor
function convertDatasetForExtraction(dataset)
    for i = 1, dataset:size() do
        dataset[i][1] = dataset[i][1]:double()
        dataset[i][2] = dataset[i][2]:double()
    end
end






params = Input:parseParams()

if (params.json ~= '') then
    local modelParams = ModelBuilder:readModel(params.json)
    
    Logger:open(modelParams.logging.logFile)
    
    Logger:info("\n----- Starting run -----")
    Logger:info("build = " .. params.build)
    Logger:info("train = " .. params.train)
    Logger:info("extract = " .. params.extract)
    Logger:info("contribs = " .. params.contribs)
    Logger:info("activations = " .. params.activations)
    
    -- Set whether GPU is used, then uses CudaTensors instead of default
    if (modelParams.training.gpu) then
        Utils.cuda = true
    end
        
    -- load data if going to train
    local n, m, dataset, genset, clusters = nil, nil, nil, nil, nil
    
    if (params.train == 'yes' or params.build == 'yes' or params.activations == 'yes') then
        Logger:info("Loading dataset from file " .. modelParams.data.input)
        n, m, dataset, genset = DataLoader:load(modelParams.data.input, modelParams.data.generalization)
        if (dataset ~= nil) then
            Logger:info("Loaded dataset with dimensions " .. n .." x " .. m)
        else
            Logger:error("Failed to load dataset, exiting")
            os.exit(1)
        end
        
        -- load clusters file if exists
        if (modelParams.data.known_clusters ~= nil) then
            clusters = DataLoader:loadKnownClusters(modelParams.data.known_clusters)
        else
            Logger:info("known_clusters is nil, not loading file")
        end
    end
    
    -- Build model if necessary
    -- NOTE: a model cannot be build unless data is loaded
    --      BUT you can load an existing model without data
    -- TODO: if building a model to train on GPU I don't know whether the saved model 
    --       on disk will include GPU info, might need to re-cast after loading
    local model, model_l1, model_l2, layerMap = nil, nil, nil, nil
    local itrsCompleted = 0
    if (params.build == 'yes') then
        Logger:info("Building model from network structure in model_format file")
        model = ModelBuilder:buildModel(n, m, modelParams.training.gpu)
        layerMap = ModelBuilder:getLayerMap()
        --model = ModelBuilder:buildModelDeep(n, m)
    else
        local modelFile = ""
        if (string.find(modelParams.data.model, ".txt") ~= nil) then
            -- model param is file
            modelFile = modelParams.data.model
        else
            -- model param is dir
            Logger:info("Loading model from file with greatest number in directory " .. modelParams.data.model)
            -- find most recent file
            local f = "ls -1rv " .. modelParams.data.model .. " | head -1"
            local handle = io.popen(f)
            local result = handle:read("*a")
            result = string.gsub(result, '\n', '')
            
            modelFile = modelParams.data.model .. result
            itrsCompleted = tonumber(string.gsub(result, "itrs.txt", ""), 10)
        end
           
        print (itrsCompleted)
        layerMap = ModelBuilder:getLayerMap()
        model = torch.load(modelFile)
    end
    
    -- try to load model_l1
    local model_l1File = nil
    if (modelParams.data.model_l1 ~= nil) then
        if (string.find(modelParams.data.model_l1, ".txt") ~= nil) then
            -- model param is file
            model_l1File = modelParams.data.model_l1
        else
            -- model param is dir
            Logger:info("Loading model_l1 from file with greatest number in directory " .. modelParams.data.model_l1)
            -- find most recent file
            local f = "ls -1rv " .. modelParams.data.model_l1 .. " | head -1"
            local handle = io.popen(f)
            local result = handle:read("*a")
            result = string.gsub(result, '\n', '')
            model_l1File = modelParams.data.model_l1 .. result
        end
        model_l1 = torch.load(model_l1File)
    end
    
    
    -- try to load model_l2
    local model_l2File = nil
    if (modelParams.data.model_l2 ~= nil) then
        if (string.find(modelParams.data.model_l2, ".txt") ~= nil) then
            -- model param is file
            model_l2File = modelParams.data.model_l2
        else
            -- model param is dir
            Logger:info("Loading model_l2 from file with greatest number in directory " .. modelParams.data.model_l2)
            -- find most recent file
            local f = "ls -1rv " .. modelParams.data.model_l2 .. " | head -1"
            local handle = io.popen(f)
            local result = handle:read("*a")
            result = string.gsub(result, '\n', '')
            model_l2File = modelParams.data.model_l2 .. result
        end
        model_l2 = torch.load(model_l2File)
    end
    
    print ("model = ", model)
    print ("model_l1 = ", model_l1)
    print ("model_l2 = ", model_l2)
    print ("layer map = ", layerMap)
    
    -- Load word2vec mapping and wordDict
    local word2vec, wordDict = nil, nil
    if (params.train == 'yes' or params.extract == 'yes') then
        if (modelParams.features.word2vecmap ~= nil) then
            word2vec = DataLoader:loadWord2VecMapping(modelParams.features.word2vecmap)
        end
        if (modelParams.data.words ~= nil) then
            wordDict = DataLoader:loadWordDict(modelParams.data.words)
        end
    end

    -- Now actually train
    if (params.train == 'yes') then
        Logger:info("Training model on data")
        Logger:info("    Learning rate = " .. modelParams.training.lr)
        Logger:info("    Total iterations = " .. modelParams.training.itrs)
        Logger:info("    Model save iterations = " .. modelParams.logging.modelSaveItrs)
        Logger:info("    Contribs save iterations = " .. modelParams.logging.contribsSaveItrs)
        Logger:info("    Features save iterations = " .. modelParams.logging.featuresSaveItrs)
        Logger:info("    Using GPU? = " .. tostring(modelParams.training.gpu))
        Logger:info("    Trainer = " .. modelParams.training.trainer)
        Logger:info("    Criterion = " .. modelParams.training.criterion)
        
        local trainer = ModelBuilder:getTrainer(model, modelParams.training.gpu)
        --if (modelParams.gpu) then
        --    convertDatasetForCuda(dataset)
        --end
        
        trainer.learningRate = modelParams.training.lr
        trainer.verbose = false
        trainAutoencoder(model,
                         model_l1,
                         model_l2,
                         trainer, 
                         itrsCompleted,
                         params,
                         modelParams, 
                         dataset,
                         clusters,
                         word2vec,
                         wordDict)
        --if (modelParams.training.gpu) then
        --    convertDatasetForExtraction(dataset)
        --end
        --trainAutoencoderCudaTest(model, trainer, tonumber(modelParams.logging.saveItrs), tonumber(modelParams.training.itrs), modelParams.data.model, dataset)
    end
    
    
    -- find hidden layer activations
    if (params.activations == 'yes') then
        --                         push through layer 1
        getLayerActivations(model, 1, dataset, modelParams.data.activations .. "default.txt")
    end
    
    if (params.contribs == 'yes') then
        if (modelParams.data.contribs ~= nil) then
            determineContributions(model, modelParams.data.contribs .. "default.txt", model_l1, model_l2)
        end
    end    
    
    -- extract data if desired
    if (params.extract == 'yes' and modelParams.data.words ~= '' and modelParams.data.features ~= '') then
        if (wordDict ~= nil or word2vec ~= nil) then
            extractModelData(model, wordDict, word2vec, modelParams.data.features .. "default.txt")         
        end
    end
        
end

