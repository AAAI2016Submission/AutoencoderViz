
Utils = {}

Utils.cuda = false

-- Returns a dense torch.Tensor for output pattern
-- elems is a tensor of form { {k, v}, {k, v}, ... }
-- size = the number of elements in the Tensor 
Utils.getDenseTensorFromSparseTensor = function(tbl, elems, size)
    local dense = nil
    
    if (Utils.cuda) then
        dense = torch.CudaTensor(size):zero()
    else 
        dense = torch.Tensor(size):zero()
    end
    
    --  set the non-zero values
    for i = 1, elems:size(1) do
        local pair = elems[i]
        dense[pair[1]] = pair[2]
    end    
    
    return dense
end


-- Returns a dense torch.Tensor for output pattern
-- elems is a table of form { {k, v}, {k, v}, ... }
-- size = the number of elements in the Tensor 
Utils.getDenseTensorFromTable = function(tbl, elems, size)
    local dense = nil
    
    if (Utils.cuda) then
        dense = torch.CudaTensor(size):zero()
    else 
        dense = torch.Tensor(size):zero()
    end
        
    --  set the non-zero values
    for key, value in pairs(elems) do
        dense[key] = value
    end
    
    return dense
end



-- Returns a sparse vector representation for input into
-- a SparseLinear layer
-- Note that the nzelem must be 1 indexed
-- TODO improve this to handle multiple non-zero elements
Utils.getSparseTensor = function(tbl, nzelem, nzval)
    local p = nil
    
    if (Utils.cuda) then
        p = torch.CudaTensor(1,2)
    else 
        p = torch.Tensor(1,2)
    end
    
    p[1][1] = nzelem -- The index that will be 1. 
    p[1][2] = nzval
    
    return p
end


