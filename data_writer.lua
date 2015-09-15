
DataWriter = {}


DataWriter.featuresFile = nil
DataWriter.activationsFile = nil
DataWriter.contribsFile = nil


-- Open a file to write features to
DataWriter.openFeaturesFile = function(tbl, path, numPats, numFeatures)
    if (DataWriter.featuresFile ~= nil) then
        return DataWriter.featuresFile
    end
    
    DataWriter.featuresFile = io.open(path, 'w')
    DataWriter.featuresFile:write(tostring(numPats), ' ', tostring(numFeatures), '\n')
end

DataWriter.closeFeaturesFile = function(tbl)
    if (DataWriter.featuresFile ~= nil) then
        DataWriter.featuresFile:close()
    end
    
    DataWriter.featuresFile = nil
end


-- Open file to write activations to
DataWriter.openActivationsFile = function(tbl, path, numPats, numFeatures)
    if (DataWriter.activationsFile ~= nil) then
        return DataWriter.activationsFile
    end
    
    DataWriter.activationsFile = io.open(path, 'w')
    DataWriter.activationsFile:write(tostring(numPats), ' ', tostring(numFeatures), '\n')
end

DataWriter.closeActivationsFile = function(tbl)
    if (DataWriter.activationsFile ~= nil) then
        DataWriter.activationsFile:close()
    end
    
    DataWriter.activationsFile = nil
end


-- Open file to write contribs to
-- This write the contributions in the format
-- 
-- index wordInd1:val1,wordInd2:val2,...,wordIndn:valn
--
DataWriter.openContribsFile = function(tbl, path)
    if (DataWriter.contribsFile ~= nil) then
        return DataWriter.contribsFile
    end
    
    DataWriter.contribsFile = io.open(path, 'w')
end

DataWriter.closeContribsFile = function(tbl)
    if (DataWriter.contribsFile ~= nil) then
        DataWriter.contribsFile:close()
    end
    
    DataWriter.contribsFile = nil
end




-- write the features to the file
-- features is a list
-- each feature is on a new line and there is a linebreak between words
DataWriter.writeFeatures = function(tbl, word, featureVecs)
    if (DataWriter.featuresFile == nil) then
        print ("Must open file first")
        return
    end
    
    DataWriter.featuresFile:write(word .. ' ')
    
    -- for each feature vec
    for i = 1, table.getn(featureVecs) do
        -- for each feature
        local feature = featureVecs[i]
        for j = 1, feature:size(1) do
            DataWriter.featuresFile:write(tostring(feature[j]))
            if (j ~= feature:size(1)) then
                DataWriter.featuresFile:write(',')
            end
        end
        DataWriter.featuresFile:write(' ')
    end
    
    DataWriter.featuresFile:write('\n')
end

--
-- activation is a tensor
DataWriter.writeActivation = function(tbl, activation)
    if (DataWriter.activationsFile == nil) then
        print ("Must open file first")
        return
    end
        
    -- for each feature vec
    for i = 1, activation:size(1) do
        DataWriter.activationsFile:write(tostring(activation[i]))
        DataWriter.activationsFile:write(' ')
    end
    
    DataWriter.activationsFile:write('\n')
end


-- Convert everything here that is 1 indexed to 0 indexed
DataWriter.writeContrib = function(tbl, hiddenNode, contrib) 
    local line = tostring(hiddenNode-1) .. " "
    
    local sz = contrib:size(1)
    for i = 1, sz do
        line = line .. tostring(i-1) .. ":" .. contrib[i]
        if (i ~= sz) then
            line = line .. ","
        end
    end
    line = line .. '\n'
    DataWriter.contribsFile:write(line)
end



