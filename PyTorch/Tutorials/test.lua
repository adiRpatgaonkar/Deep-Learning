require 'torch'
require 'nn'
require 'paths'
require 'itorch'

if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end

train_data = torch.load('cifar10-train.t7')
test_data = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(#train_data.data)
print(train_data.data[100])
print(classes[train_data.labels[100]])

-- Return the index of the data set example & its label]]--
setmetatable(train_data, {_index = function(t, i)
                                        return {t.data[i], t.label[i]}
                                    end}
);
train_data.data = train_data.data:double()

-- Return the size of the data set
function train_data:size()
    return self.data:size(1)
end

-- Test the above functions
print(train_data:size())
print(train_data.data[33])

-- Index operator

red_channel = train_data.data[{ {}, {1}, {}, {} }] -- A
print(#red_channel)
index_exmp = train_data.data[{ {1, 300}, {}, {}, {} }]
print(#index_exmp)

net_cifar = nn.Sequential()
net_cifar:add(nn.SpatialConvolution(3, 4, 5, 5))
net_cifar:add(nn.ReLU())
net_cifar:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net_cifar:add(nn.SpatialConvolution(6, 14, 5, 5))
net_cifar:add(nn.ReLU())
net_cifar:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net_cifar:add(nn.View(16 * 5 * 5))
net_cifar:add(nn.Linear(16 * 5 * 5, 120))
net_cifar:add(nn.ReLU())
net_cifar:add(nn.Linear(120, 84))
net_cifar:add(nn.ReLU())
net_cifar:add(nn.Linear(84 ,10))
net_cifar:add(nn.LogSoftMax())
print(net_cifar)
ce_criterion = nn.ClassNLLCriterion()

horse = testing_set.data[100]
print(horse:mean(), horse:std())