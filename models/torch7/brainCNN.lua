-- Aly Valliani and Andrew Gilchrist-Scott
-- March 21, 2016
--
-- MRI counts:
--      197 AD
--      220 CN
--      158 MCI
--      136 LMCI
--
-- Activate Torch: . /usr/local/torch/install/bin/torch-activate
-- Usage: qlua brainCNN.lua (alternatively, th brainCNN_1.lua, but this
-- won't graphically display the confusion matrix)


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

ADTotal = 0
CNTotal = 0
-- ADAvg = 0

-- Load AD
fileAD = io.open('/sonigroup/bl_ADNI/AD/filenamesAD.txt', 'r')
for i=1, 197 do
  images[i] = npy4th.loadnpy('/sonigroup/bl_ADNI/AD/' .. fileAD:read())
  -- ADAvg = ADAvg + images[i]:mean()
  labels[i] = 0
end
-- print(ADAvg/197)

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
  if trainSet.labels[i] == 0 then
    ADTotal = ADTotal + trainSet.data[i]:mean()
  else
    CNTotal = CNTotal + trainSet.data[i]:mean()
  end
end

for i=trainSize+1, trainSize+testSize do
  --testSet.data[i-trainSize] = images[labelsShuffle[i]][1]:clone()
  --testSet.labels[i-trainSize] = labels[labelsShuffle[i]]
  testSet.data[i-trainSize] = images[permList[i]][1]:clone()
  testSet.labels[i-trainSize] = labels[permList[i]]
  if testSet.labels[i-trainSize] == 0 then
    ADTotal = ADTotal + testSet.data[i-trainSize]:mean()
  else
    CNTotal = CNTotal + testSet.data[i-trainSize]:mean()
  end
end

print("***AD Avg Image Activation***")
print(ADTotal/197)
print("***CN Avg Image Activation***")
print(CNTotal/220)

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

net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 128, 11, 11))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(7, 7, 3, 3))

net:add(nn.SpatialConvolution(128, 64, 11, 11))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(5, 5, 2, 2))
net:add(nn.Dropout(0.1))

net:add(nn.View(64*33*18))
net:add(nn.Linear(64*33*18, 500))
net:add(nn.ReLU())
net:add(nn.Linear(500, 100))
net:add(nn.ReLU())
net:add(nn.Linear(100, 50))
net:add(nn.ReLU())
net:add(nn.Linear(50, 20))
net:add(nn.ReLU())
net:add(nn.Linear(20, 2))
net:add(nn.LogSoftMax())


criterion = nn.ClassNLLCriterion() -- Use log-likelihood classification loss
confusionMatrix = optim.ConfusionMatrix(classes)

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.weightDecay = 0.001
trainer.maxIteration = 1 -- 5 epochs of training

-- Transfer to GPU
net = net:cuda()
criterion = criterion:cuda()
trainSet.data = trainSet.data:cuda()
testSet.data = testSet.data:cuda()

-- Train the network
trainer:train(trainSet)

-- Visualization of weights (requires itorch)
if itorch then
  itorch.image(net.weight)
end

for i=1, 84 do
  if testSet.labels[i] == 0 then
    res = net:forward(testSet.data[i])
    break
  end
end
print(res:sum())

-- Determine accuracy
correct = 0 
for i=1, 84 do
  local truth = testSet.labels[i]
  local prediction = net:forward(testSet.data[i])
  local confidences, indices = torch.sort(prediction, true)
  if truth == indices[1] then
    correct = correct + 1
  end
  confusionMatrix:add(prediction, truth)
end

print(correct, 100 *correct/84 .. ' % ')
print(confusionMatrix)
--image.display(confusionMatrix:render())

-- print('Weights:')
-- print(net:get(4).weight)

-- if itorch then
  -- print '==> Visualizing ConvNet filters'
  -- print('Layer 1 filters:')
  -- itorch.image(net:get(1).weight)
-- end


