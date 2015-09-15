
Logger = {}

Logger.file = nil

-- Log levels
Logger.INFO = "INFO"
Logger.DEBUG = "DEBUG"
Logger.WARN = "WARN"
Logger.ERROR = "ERROR"
 
Logger.open = function(tbl, path)
    if (Logger.file == nil) then
        -- if the file has not already been initialized
        Logger.file = io.open(path, 'a')
    end
end


Logger.info = function(tbl, msg) 
    if (Logger.file ~= nil) then
        Logger.file:write(Logger.INFO .. ' [' .. os.time() .. ']:\t' .. msg .. '\n')
        Logger.file:flush()
    else
        print ("Logger file is nil")
    end
end


Logger.debug = function(tbl, msg)
    if (Logger.file ~= nil) then
        Logger.file:write(Logger.DEBUG .. ' [' .. os.time() .. ']:\t' .. msg .. '\n')
        Logger.file:flush()
    else
        print ("Logger file is nil")
    end
end

Logger.warn = function(tbl, msg)
    if (Logger.file ~= nil) then
        Logger.file:write(Logger.WARN .. ' [' .. os.time() .. ']:\t' .. msg .. '\n')
        Logger.file:flush()
    else
        print ("Logger file is nil")
    end
end

Logger.error = function(tbl, msg)
    if (Logger.file ~= nil) then
        Logger.file:write(Logger.ERROR .. ' [' .. os.time() .. ']:\t' .. msg .. '\n')
        Logger.file:flush()
    else
        print ("Logger file is nil")
    end
end


Logger.close = function(tbl) 
    if (Logger.file ~= nil) then
        -- if the file has been initialized, close it
        Logger.file:close()
    end
end
