
require 'nn'
local luajson = require('cjson')

ModelBuilder = {}
ModelBuilder.json = nil


--
ModelBuilder.readModel = function(tbl, path)
    local file = io.open(path, "rb")
    local content = file:read("*all")
    file:close()
    
    local json = luajson.decode(content)
    
    -- Holds relevant info for training purposes
    local modelParams = {}
    modelParams.logging = {}
    modelParams.data = {}
    modelParams.training = {}
    modelParams.features = {}
    
    local logging = json['logging']
    modelParams.logging.logFile = logging['file']
    modelParams.logging.modelSaveItrs = logging['model_save_itrs']
    modelParams.logging.contribsSaveItrs = logging['contribs_save_itrs']
    modelParams.logging.featuresSaveItrs = logging['features_save_itrs']
    modelParams.logging.hiddenActivationsSaveItrs = logging['hidden_activations_save_itrs']
    
    local data = json['data']
    modelParams.data.words = data['words']
    modelParams.data.features = data['features']
    modelParams.data.input = data['input']
    modelParams.data.output = data['output']
    modelParams.data.model = data['model']
    modelParams.data.model_l1 = data['model_l1']
    modelParams.data.model_l2 = data['model_l2']
    modelParams.data.activations = data['hidden_activations']
    modelParams.data.contribs = data['contribs']
    modelParams.data.known_clusters = data['known_clusters']
    -- This is the percentage to make the generalization set
    modelParams.data.generalization = data['generalization'] 

    local training = json['training']
    modelParams.training.gpu = training['gpu']
    modelParams.training.itrs = training['iterations']
    modelParams.training.lr = training['lr']
    modelParams.training.trainer = training['trainer']
    modelParams.training.criterion = training['criterion']
    
    local features = json['features']
    modelParams.features.word2vecmap = features['word2vecmap']
    
    ModelBuilder.json = json
        
    return modelParams
end



ModelTypes = {}

ModelTypes.LayerSparseLinear = "SparseLinear"
ModelTypes.LayerLinear = "Linear"
ModelTypes.TransferTanh = "Tanh"
ModelTypes.TrainerSGD = "SGD"
ModelTypes.CriterionMSE = "MSE"



ModelBuilder.getLayerFromType = function(tbl, ltype, n_input, n_output)
    if (ltype == ModelTypes.LayerSparseLinear) then
        return nn.SparseLinear(n_input, n_output)
    elseif (ltype == ModelTypes.LayerLinear) then
        return nn.Linear(n_input, n_output)
    end
end


ModelBuilder.getTransferFromType = function(tbl, ttype)
    if (ttype == ModelTypes.TransferTanh) then
        return nn.Tanh()
    end
end


ModelBuilder.model = nil
ModelBuilder.lastLayerAdded = nil
ModelBuilder.numLayers = nil -- number of layer blocks, actual layers is one more
ModelBuilder.layerMap = nil -- keeps track of the output index for each layer


ModelBuilder.getModel = function()
    return ModelBuilder.model
end



ModelBuilder.getLayerMap = function()
    if (ModelBuilder.layerMap == nil) then
        ModelBuilder:buildLayerMap()
    end
    return ModelBuilder.layerMap
end


-- Generates layer map without building model and without requiring data dimension
ModelBuilder.buildLayerMap = function(tbl)
    local network = ModelBuilder.json['network']
    local layers = network['layers']
    
    ModelBuilder.layerMap = {}   
    local numModules = 0
    
    for i = 1, table.getn(layers) do
        local layer = layers[i]
        
        -- The base layer (i.e. Linear, SparseLinear, etc)
        numModules = numModules + 1
        
        -- Does this layer have dropout?
        if (layer['dropout'] ~= nil) then
            numModules = numModules + 1
        end   
        
        -- Does this layer have a transfer function?
        if (layer['transfer'] ~= nil) then
            numModules = numModules + 1
        end
        
        ModelBuilder.layerMap[i] = numModules
    end

    return model
end


-- Build the model from the network structure in the file
ModelBuilder.buildModel = function(tbl, n, m, gpu)
    local network = ModelBuilder.json['network']
    local layers = network['layers']
    
    local model = nn.Sequential()
    
    ModelBuilder.layerMap = {}
    local numModules = 0

    -- copy to CudaTensor
    --if (gpu) then
    --    model:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
    --    print ("Adding copy to cudatensor")
    --    numModules = numModules + 1
    --end
        
    for i = 1, table.getn(layers) do
        local layer = layers[i]
        local type = layer['type']
        
        local n_input = layer['n_input']
        local n_output = layer['n_output']
        if (n_input == "data") then
            n_input = m
        end
        if (n_output == "data") then
            n_output = m
        end
        
        model:add(ModelBuilder:getLayerFromType(type, n_input, n_output))
        numModules = numModules + 1
        
        -- check for dropout and sparsity
        if (layer['dropout'] ~= nil) then
            model:add(nn.Dropout(layer['dropout']))
            numModules = numModules + 1
        end   
        
        -- Add transfer
        if (layer['transfer'] ~= nil) then
            model:add(ModelBuilder:getTransferFromType(layer['transfer']))
            numModules = numModules + 1
        end
        
        ModelBuilder.layerMap[i] = numModules
    end

    -- if want to train on the GPU, re-cast model for CUDA
    -- http://hunch.net/~nyoml/torch7.pdf
    if (gpu) then
        model:cuda()
        print ("re-casting model for CUDA")
    end
    
    return model
end




ModelBuilder.getTrainer = function(tbl, model, gpu)
    Logger:info("Building trainer from model and criterion")
    
    local training = ModelBuilder.json['training']
    local criterion = nil
    local trainer = nil
    
    if (training['criterion'] == "MSE") then
        criterion = nn.MSECriterion()
        if (gpu) then
            criterion = criterion:cuda()
        end
    end
    
    if (training['trainer'] == "SGD") then
        trainer = nn.StochasticGradient(model, criterion, gpu)
    end
    
    return trainer
end



-----
-----
-----
-- Functions for deep model building
-----
-----
-----
ModelBuilder.hasNextLayer = function() 
    return ModelBuilder.lastLayerAdded ~= ModelBuilder.numLayers
end


ModelBuilder.addNextLayer = function()
    -- 
    -- set previous layers to static
    -- remove output layer
    -- Add layer[ModelBuilder.lastLayerAdded] as autoencoder
    --
    print ("Adding next layer to model")
    
    local l = ModelBuilder.lastLayerAdded
    ModelBuilder.model:remove()
    local layers = ModelBuilder.json['network']['layers']
    -- Add lth layer and (l+1)th layer with output as input to lth layer
    ModelBuilder:addAutoencoderBlock(layers, ModelBuilder.lastLayerAdded)    
end


ModelBuilder.addAutoencoderBlock = function(layers, l) 
    -- Add the autoencoder starting at layer l, then add the next layer with 
    -- output same as input to train the new hidden layer
    for i = 0, 1 do
        local layer = layers[i+l]
        local type = layer['type']
        
        local n_input = layer['n_input']
        local n_output = layer['n_output']
        
        if (i == 2) then
            -- set the output to be the input size of previous layer
            n_output = layers[1]['n_input']
        end
        
        if (n_input == "data") then
            n_input = m
        end
        if (n_output == "data") then
            n_output = m
        end
        
        ModelBuilder.model:add(ModelBuilder:getLayerFromType(type, n_input, n_output))
        
        -- check for dropout and sparsity
        if (layer['dropout'] ~= nil) then
            ModelBuilder.model:add(nn.Dropout(layer['dropout']))
        end   
        
        -- Add transfer
        if (layer['transfer'] ~= nil) then
            ModelBuilder.model:add(ModelBuilder:getTransferFromType(layer['transfer']))
        end

    end
    
    ModelBuilder.lastLayerAdded = ModelBuilder.lastLayerAdded + 1

end


ModelBuilder.buildModelDeep = function(tbl, n, m) 
    -- read json layers
    -- create model as if only 3 layers
    -- 
    local network = ModelBuilder.json['network']
    local layers = network['layers']
    
    ModelBuilder.model = nn.Sequential()
    ModelBuilder.numLayers = table.getn(layers)
    print ("Deep autoencoder has ", ModelBuilder.numLayers, "layers")
    
    -- Build first autoencoder
    ModelBuilder:addAutoencoderBlock(layers, 1)
    
    return ModelBuilder.model
end


