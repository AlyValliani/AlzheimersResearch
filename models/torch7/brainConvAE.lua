require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'cunn'
require 'cutorch'
npy4th = require 'npy4th'
require 'math'

-- Create image and label tensors
n = 417
images = torch.Tensor(417, 1, 256, 166)
labels = torch.Tensor(417)
classes = {0, 1}

-- Load AD
fileAD = io.open('/sonigroup/bl_ADNI/AD/filenamesAD.txt', 'r')
for i=1, 197 do
  images[i] = npy4th.loadnpy('/sonigroup/bl_ADNI/AD/' .. fileAD:read())
  labels[i] = 0
end

-- Load CN
fileCN = io.open('/sonigroup/bl_ADNI/CN/filenamesCN.txt', 'r')
for i=198, 417 do
  images[i] = npy4th.loadnpy('/sonigroup/bl_ADNI/CN/' .. fileCN:read())
  labels[i] = 1
end

-- Shuffle the data set
--labelsShuffle = torch.randperm((#labels)[1])

--Generate permutation list
permList = {}
for i = 1, n do
  permList[i] = i
end

for i = 1, n do
  j = math.random(i, n)
  permList[i], permList[j] = permList[j], permList[i]
end

portionTrain = 0.8 -- 80% train data, 20% test
--trainSize = torch.floor(labelsShuffle:size(1)*portionTrain)
--testSize = labelsShuffle:size(1) - trainSize
trainSize = torch.floor(n*portionTrain)
testSize = n - trainSize

-- Create training set
trainSet = {
  data = torch.Tensor(trainSize, 1, 256, 166),
  labels = torch.Tensor(trainSize),
  size = function() return trainSize end
}

-- Create testing set
testSet = {
  data = torch.Tensor(testSize, 1, 256, 166),
  labels = torch.Tensor(testSize),
  size = function() return testSize end
}

for i=1, trainSize do
  --trainSet.data[i] = images[labelsShuffle[i]][1]:clone()
  --trainSet.labels[i] = labels[labelsShuffle[i]]
  trainSet.data[i] = images[permList[i]][1]:clone()
  trainSet.labels[i] = labels[permList[i]]
end

for i=trainSize+1, trainSize+testSize do
  --testSet.data[i-trainSize] = images[labelsShuffle[i]][1]:clone()
  --testSet.labels[i-trainSize] = labels[labelsShuffle[i]]
  testSet.data[i-trainSize] = images[permList[i]][1]:clone()
  testSet.labels[i-trainSize] = labels[permList[i]]
end

-- Set index operator
setmetatable(trainSet,
        {__index = function(t, i)
                                        return {t.data[i], t.labels[i]}
                                end}
);

-- Convert from ByteTensor to DoubleTensor
trainSet.data = trainSet.data:double()
testSet.data = testSet.data:double()

-- Remove temporary files from memory
images = nil
labels = nil

conntable = nn.tables.full(1, 16)
kw, kh = 11, 11
iw, ih = 256, 166

-- connection table:
local decodertable = conntable:clone()
decodertable[{ {}, 1 }] = conntable[{ {}, 2 }]
decodertable[{ {}, 2 }] = conntable[{ {}, 1 }]
local outputFeatures = conntable[{ {}, 2 }]:max()

-- encoder:
encoder = nn.Sequential()
encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(outputFeatures))

-- decoder:
decoder = nn.Sequential()
decoder:add(nn.SpatialFullConvolutionMap(decodertable, kw, kh, 1, 1))

-- complete model:
module = unsup.AutoEncoder(encoder, decoder, 1)

criterion = nn.ClassNLLCriterion() -- Use log-likelihood classification loss
confusionMatrix = optim.ConfusionMatrix(classes)

trainer = nn.StochasticGradient(module, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 1 -- 5 epochs of training

-- Transfer to GPU
module = module:cuda()
criterion = criterion:cuda()
trainSet.data = trainSet.data:cuda()
testSet.data = testSet.data:cuda()

-- Train the network
trainer:train(trainSet)

