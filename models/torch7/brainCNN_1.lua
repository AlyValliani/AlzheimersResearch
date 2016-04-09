-- This is the initial CNN constructed to test whether brain MRIs
-- stored as npy files are being loaded appropriately.
--
-- Aly Valliani and Andrew Gilchrist-Scott
-- February 13, 2016
--
-- Activate Torch: . /usr/local/torch/install/bin/torch-activate
-- Usage: qlua brainCNN_1.lua (alternatively, th brainCNN_1.lua, but this
-- won't graphically display the confusion matrix)

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
npy4th = require 'npy4th'

print("Loading npy array into Torch tensor...")
array = npy4th.loadnpy('/sonigroup/bl_ADNI/AD/ssrADNI_141_S_1024_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070402181417383_S22699_I47748_cut.npy')
classes = {'AD', 'CN', 'MCI'}
print("...Completed loading")

array = array:double()
print(array:size()) --256x166

print("Converting Torch tensor into table for training...")
local table = {}
for i=1, array:size(1) do
  table[i] = {}
  for j=1, array:size(2) do
    table[i][j] = array[i][j]
  end
end
print("...Completed conversion")

print(table[200][100])

-- Set index operator
setmetatable(table,
	{__index = function(t, i)
					return {t.data[i], t.label[i]}
				end}
);

--function table:size()
--	return self:size(2)
--end

--print(table:size())

net = nn.Sequential()
criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 1

-- Transfer to GPU
-- net = net:cuda()
-- criterion = criterion:cuda()
-- table = table:cuda()

print("Training...")
trainer:train(table)
print("...Training Complete")
