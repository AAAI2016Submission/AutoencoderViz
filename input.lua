
Input = {}

Input.parseParams = function()
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Training a simple autoencoder')
    cmd:text()
    cmd:text('Options')

    -- general options:
    cmd:option('-json', '', 'json model file')
    cmd:option('-build', 'no', 'whether to build model from network definition or use existing one, yes or no')
    cmd:option('-train', 'no', 'whether to train on this model, yes or no')
    cmd:option('-extract', 'no', 'whether to write features, yes or no')
    cmd:option('-contribs', 'no', 'whether to compute hidden layer contributions, yes or no')
    cmd:option('-activations', 'no', 'write hidden layer activations, yes or no')

    cmd:text()

    params = cmd:parse(arg)
    return params
end
