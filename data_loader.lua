
require 'utils'
require 'logger'

-- This file will convert all arrays that are 0 indexed on disk to
-- 1 indexed in memory

DataLoader = {}

DataLoader.TYPE_1ON = 0
DataLoader.TYPE_BOW = 1

DataLoader.isSparse = false

DataLoader.getType = function(tbl, val) 
    if (val == DataLoader.TYPE_1ON) then
        return "1ON"
    elseif (val == DataLoader.TYPE_BOW) then
        return "BOW"
    else
        return "Unknown Type"
    end
end


-- load the words into a dictionary
DataLoader.loadWordDict = function(tbl, path)

    Logger:info("Loading worddict from file " .. path)

    local wordDict = {}
    local file = io.open(path, 'r')

    for line in file:lines() do    
        local count = 1    
        local word = ""
        local val = -1
        for i in string.gmatch(line, "%S+") do
            if (count == 1) then
                word = i
            elseif (count == 2) then
                val = tonumber(i) + 1 -- The data from Python is 0-indexed
            end
            count = count + 1
        end
        
        wordDict[word] = val
    end
    
    file:close()    
    return wordDict    
end


DataLoader.loadKnownClusters = function(tbl, path)

    Logger:info("Loading known clusters from file " .. path)
    
    local clusters = {}
    local file = io.open(path, 'r')
    
    for line in file:lines() do
        local cluster = {}
        for w in string.gmatch(line, "([^,]+)") do
            cluster[#cluster + 1] = w
        end
        clusters[#clusters + 1] = cluster
    end
    
    file:close()
    
    return clusters
end


DataLoader.loadWord2VecMapping = function(tbl, path)

    Logger:info("Loading word2vec mapping from file " .. path)
    
    local word2vec = {}
    local file = io.open(path, 'r')
    
    local dimsLine = file:read()
    local pats = nil
    local features = nil
    
    for d in string.gmatch(dimsLine, "%S+") do
        if (pats == nil) then
            pats = tonumber(d)
        else
            features = tonumber(d)
        end
    end
    
    for line in file:lines() do
        local word = nil
        local vecStr = nil
        
        -- split line into word and vec 
        for i in string.gmatch(line, "%S+") do
            if (word == nil) then
                word = i
            else
                vecStr = i
            end
        end
        
        vec = torch.Tensor(features)
        -- split vector into values
        local count = 1
        for i in string.gmatch(vecStr, "([^,]+)") do
            --print (i)
            vec[count] = tonumber(i)
            count = count + 1
        end
        
        word2vec[word] = vec
    end
    
    file:close()
    return word2vec
end



DataLoader.load = function(tbl, path, generalization)
    
    -- Check if file is txt or h5
    local ext = string.match(path, "([^%.]+)$")
    local n, m, dataset = nil, nil, nil
    if (ext == 'h5') then
        Logger:info("Loading data from HDF5 format")
        local file = hdf5.open(path, 'a')
        
        -- 1on or bow?
        local type = file:read('/meta/type'):all()
        Logger:info("Data is in " .. DataLoader:getType(type[1]) .. " format")
        
        -- Is the data sparse?
        local sparse = file:read('/meta/sparse'):all()
        DataLoader.isSparse = (sparse[1] == 1)
        Logger:info("Data is sparse? = " .. tostring(DataLoader.isSparse))
        
        if (sparse[1] == 1) then
            -- Sparse
            if (type[1] == DataLoader.TYPE_1ON) then
                n, m, dataset = DataLoader:loadHDF5SparseOneOfN(file)
            else
                n, m, dataset = DataLoader:loadHDF5SparseBOW(file)
            end
        else
            -- Dense, doesn't matter the format. 
            -- It's just a matrix
            n, m, dataset = DataLoader:loadHDF5Dense(file)
        end
    elseif (ext == 'txt') then
        Logger:info("Loading data from text format")
        n, m, dataset = DataLoader:loadDatasetText(path)
    end
    
    if (n ~= nil and m ~= nil and dataset ~= nil) then
        local shuffled = torch.randperm(n, 'torch.LongTensor')
        local gensize = torch.floor(n * generalization)
        local genset = {}
        for i = 1, gensize do
            -- move dataset[shuffled[i]] to genSet
            genset[i] = dataset[shuffled[i]]
            dataset[shuffled[i]] = nil
        end
        
        -- Remove any values set to nil
        for j = n, 1, -1 do 
            if (dataset[j] == nil) then
                table.remove(dataset, j)
            end
        end
        
        function dataset:size() return n - gensize end
        function genset:size() return gensize end

        Logger:info("Dataset size = " .. dataset:size())
        Logger:info("Generalization set size = " .. genset:size())
        
        return n, m, dataset, genset
    else
        Logger:error("Dataset is nil")
    end

end



-- Will load sparse format for 1 of N encoding
-- This assumes the first layer of the network will be
-- SparseLinear and the output is [Dense]Linear 
DataLoader.loadHDF5SparseOneOfN = function(tbl, file) 
    
    Logger:info("Loading HDF5 sparse 1 of N")
    
    local dims = file:read('/meta/dim'):all()
    
    -- The array is a list of positions of ones
    local array = file:read('/data/matrix'):all()
    
    -- n is the number of patterns
    -- m is the number of elements in each pattern
    n = dims[1]
    m = dims[2]
    
    -- create n x 2 table where n is the number of patterns
    local dataset = {}
    
    for i = 1, n do
        -- Create tensor of (1 x 2) since there will always be only one non-zero index
        elem = array[i] + 1 -- Add 1 because data is written 0-indexed
        dataset[i] = {Utils:getSparseTensor(elem, 1), Utils:getDenseTensor({elem, 1.0}, m)}
    end
            
    file:close()
    return n, m, dataset

end


-- Loads data for BOW given CSR sparse format
-- TODO: not cuda compatible, need getSparseVector function with multiple non zero indices
DataLoader.loadHDF5SparseBOW = function(tbl, file) 
    
    Logger:info("Loading HDF5 sparse BOW")
    
    local dims = file:read('/meta/dim'):all()
    
    -- n is the number of patterns (i.e. docs)
    -- m is the number of elements in each pattern (i.e. vocab size)
    n = dims[1]
    m = dims[2]
    
    -- create n x 2 table where n is the number of patterns
    local dataset = {}
    
    -- The non zero values
    local data = file:read('/data/data'):all()
    local indices = file:read('/data/indices'):all()
    local indptr = file:read('/data/indptr'):all()

    print ("data:   ", data:size(1))
    print ("indices:", indices:size(1))
    print ("indptr: ", indptr:size(1))

    local x = 1 -- the index into the dataset, incremented only when a pattern is added
    for i = 1, n do
        -- for each document (i.e. row)
        local rowStart = indptr[i] + 1 -- Add 1 to convert from 0-indexed to 1-indexed
        local rowEnd = indptr[i + 1] + 1
        
        -- for each element in the row
        -- Build sparse table
        local nnz = rowEnd - rowStart
        --print ("nnz=",nnz)
        if (nnz == 0) then
            print ("Oh no")
            print ("s=", rowStart)
            print ("e=", rowEnd)
            print ("i=", i)
            -- if we are at an all zero pattern, decrement n
            n = n - 1
        elseif (nnz ~=0) then
            local pat = torch.Tensor(nnz, 2)
            local c = 1
            for j = rowStart, rowEnd - 1 do -- minus 1 since for loop inclusive
                pat[c][1] = indices[j] + 1 -- add 1 since indices from python are 0-indexed
                pat[c][2] = data[j]
                c = c + 1
            end        
            dataset[x] = {pat, Utils:getDenseTensorFromSparseTensor(pat, m)}
            -- if we added a pattern, increment x
            x = x + 1
        end
    end
            
    file:close() 
    return n, m, dataset
end



-- Load the full HDF5 file as a single chunk and return the dataset as a table
-- TODO: not cuda compatible
DataLoader.loadHDF5Dense = function(tbl, file)

    Logger:info("Loading dense HDF5")
    
    local matrix = file:read('/data/matrix'):all()    
    -- n is the number of patterns
    -- m is the number of elements in each pattern
    n = matrix:size(1)
    m = matrix:size(2)
    
    -- create n x 2 table where n is the number of patterns
    -- for each pattern, i, in the matrix, copy to both tensor[i][1] and tensor[i][2]
    local dataset = {}
    
    for i = 1, n do
        dataset[i] = {matrix[i], matrix[i]}
    end
        
    file:close() 
    return n, m, dataset

end



-- Load a dense dataset from a text file
DataLoader.loadDatasetText = function(tbl, path)
    
    Logger:info("Loading (dense) dataset text")
    
    local file = io.open(path, 'r')
    
    -- Find dimensions of data
    -- The python script can write in more than 2D 
    -- but this can only read 2D
    local firstLine = file:read()    
    local dims = torch.Tensor(2)
    local count = 1
    for i in string.gmatch(firstLine, "%S+") do
        if (count == 1) then
            dims[count] = tonumber(i)
        end
        if (count == 2) then
            dims[count] = tonumber(i)
        end
        count = count + 1
    end

    local dataset = {}
    local pattern = 1 
    for line in file:lines() do
    
        local kvmap = {}
        local c = 1
        for i in string.gmatch(line, "%S+") do
            local v = tonumber(i)
            if (v ~= 0) then
                kvmap[c] = v
            end
            c = c + 1
        end
        
        local d = Utils:getDenseTensorFromTable(kvmap, dims[2])

        dataset[pattern] = {d, d}
        pattern = pattern + 1
    end
    
    file:close()
    
    return dims[1], dims[2], dataset
end












--
--
-- This is currently unused -- 
--
--
-- Load the HDF5 file in chunks and return the dataset as a table
function loadDatasetChunked(tbl, file)
    print ("Loading dataset chunked")
    
    local dims = file:read('/meta/dim'):all()
    -- print ("dims: ", dims)
    
    local chunkSizes = file:read('/data/chunks'):all()
    -- print ("chunks: ", chunkSizes)
    
    -- Dimension in x,y
    local dy = dims[1]
    local dx = dims[2]
    print ("dim=", dy, dx)
    
    -- Chunk sizes in x,y dims
    local cy = chunkSizes[1]
    local cx = chunkSizes[2]
    print ("chunk sizes=", cy, cx)
    
    -- Find number of chunks and overflow into last chunk
    local ychunks = math.ceil(dy / cy)
    local xchunks = math.ceil(dx / cx)
    print ("num chunks=", ychunks, xchunks)
    
    -- create n x 2 table where n is the number of patterns
    -- for each pattern, i, in the matrix, copy to both tensor[i][1] and tensor[i][2]
    local dataset = {}
    
    -- Iterate over all chunks first going across and then down
    for y = 0, ychunks-1 do
        --print ("y=", y)
        -- Initialize cy empty tensors of length dx
        for t = 1, cy do
            dataset[y*cy+t] = {torch.Tensor(dx), torch.Tensor(dx)}
        end

        local ybegin = y * cy + 1 -- Add 1 for 1 indexed tensors
        local yend = ybegin + cy - 1 -- Subtract 1 because for loops are inclusive
        
        -- Ensure end params are in range
        if (yend >= dy) then
            yend = dy
            --print ("At end y")
        end
        
        -- Find the size of the current y chunk
        -- This is used to copy the right amount of data from the 
        -- chunks read to the datset
        -- +1 because for loops are inclusive
        local ychunkSize = yend - ybegin + 1
        --print ("ychunkSize=", ychunkSize)
        
        for x = 0, xchunks-1 do
            --print ("x=", x)
            
            local xbegin = x * cx + 1 -- Add 1 for 1 indexed tensors
            local xend = xbegin + cx - 1
            
            if (xend >= dx) then
                xend = dx
                --print ("At end x")
            end
            
            -- Find the size of the current x chunk
            -- This is used to copy the right amount of data from the 
            -- chunks read to the datset
            -- +1 because for loops are inclusive
            local xchunkSize = xend - xbegin + 1 
            --print ("xchunkSize=",xchunkSize)
            
            --print (ybegin, xbegin, yend, xend)
            
            -- that x,y are in correct places
            local data = file:read('/data/matrix'):partial({ybegin, yend}, {xbegin, xend})
            
            -- Copy data into correct location in dataset
            for a = 1, ychunkSize do
                for b = 1, xchunkSize do
                    -- Row of dataset = y * cy + a
                    -- Since we are setting up an autoencoder dataset
                    --   need to copy to both pos 1 and 2
                    -- The index in the sample = x * cx + b
                    dataset[y * cy + a][1][x * cx + b] = data[a][b]
                    dataset[y * cy + a][2][x * cx + b] = data[a][b]
                end
            end     
        end
    end
    
    -- Return the number of patterns, the length of the patterns, and the dataset table
    return dy, dx, dataset
end

