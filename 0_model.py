# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# create a model class that inherits nn.Module 这里是Module 不是model
class Model(nn.Module):
    #input layer (4 features of the flower) -->
    #  Hidden layer1 (number of neurons) -->
    #  H2(n) --> output (3 classed of iris flowers)
    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3):
        super().__init__() # instantiate out nn.Module 实例化
        self.fc1 = nn.Linear(in_features= in_features, out_features= h1)
        self.fc2 = nn.Linear(in_features= h1, out_features= h2)
        self.out = nn.Linear(in_features= h2, out_features= out_features)
    
    # moves everything forward 
    def forward(self, x):
        # rectified linear unit 修正线性单元 大于0则保留，小于0另其等于0
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    

# %%
# before we turn it on we need to create a manual seed, because networks involve randomization every time.
# say hey start here and then go randomization, then we'll get basically close to the same outputs

# pick a manual seed for randomization
torch.manual_seed(seed= 41)
# create an instance of model
model = Model()


