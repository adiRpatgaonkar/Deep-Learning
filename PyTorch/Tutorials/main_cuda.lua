require 'cunn'
require 'torch'
require 'nn'
require 'paths'
require 'image'

if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end

training_set = torch.load('cifar10-train.t7')
testing_set = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(#training_set.data)

-- Return the index of the data set example & its label]]--
setmetatable(training_set, {__index = function(t, i)
                                        return {t.data[i], t.label[i]}
                                    end}
);
training_set.data = training_set.data:double()
-- Return the size of the data set
function training_set:size()
    return self.data:size(1)
end

-- Normalize training data
mean = {}
stdev = {}

for i = 1, 3 do
	mean[i] = training_set.data[{ {}, {i}, {}, {} }]:mean()
	print('Channel '..i..', Mean '..mean[i])
	training_set.data[i]:add(-mean[i])

	stdev[i] = training_set.data[{ {}, {i}, {}, {} }]:std()
	print('Channel '..i..', Stdev '..stdev[i])
	training_set.data[i]:div(stdev[i])
end

-- Defining neural network architecture
net_cifar = nn.Sequential()
net_cifar:add(nn.SpatialConvolution(3, 12, 5, 5))
net_cifar:add(nn.ReLU())
net_cifar:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net_cifar:add(nn.SpatialConvolution(12, 24, 5, 5))
net_cifar:add(nn.ReLU())
net_cifar:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net_cifar:add(nn.View(24 * 5 * 5))
net_cifar:add(nn.Linear(24 * 5 * 5, 120))
net_cifar:add(nn.ReLU())
net_cifar:add(nn.Linear(120, 84))
net_cifar:add(nn.ReLU())
net_cifar:add(nn.Linear(84 ,10))
net_cifar:add(nn.LogSoftMax())
print(net_cifar)
ce_criterion = nn.ClassNLLCriterion()

-- For CUDA
net_cifar = net_cifar:cuda()
ce_criterion = ce_criterion:cuda()
training_set.data = training_set.data:cuda()
training_set.label = training_set.label:cuda()

training = nn.StochasticGradient(net_cifar, ce_criterion)
training.learningRate = 0.001
training.maxIteration = 50

training:train(training_set)

-- Testing starts
testing_set.data = testing_set.data:double()

for i = 1, 3 do
	testing_set.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	testing_set.data[{ {}, {i}, {}, {} }]:div(stdev[i])
end

-- For CUDA
testing_set.data = testing_set.data:cuda()
testing_set.label = testing_set.label:cuda()
predicted = net_cifar:forward(testing_set.data[100])
predicted:exp()

--[[for i = 1, predicted:size(1) do
  print(classes[i], predicted[i])
end]]--

correct = 0
for i = 1, 10000 do
	local truth = testing_set.label[i]
	local predicted = net_cifar:forward(testing_set.data[i])
	local probs, indices = torch.sort(predicted, true)
	if truth == indices[1] then
		correct = correct + 1 
	end
end

print(correct, (correct / 10000) * 100 .. ' %')