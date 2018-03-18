import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('newtestdata2.csv')
df[['Class']] = df['Class'].map({'L': 0, 'R': 1}) #map into numeric data
df = df.sample(frac=1) #shuffle dataset

m= 2475

X = torch.Tensor(np.array(df[df.columns[1:769]]))
Y = torch.Tensor(np.array(df[['Class']]))

x_train = Variable(X[:m])
y_train = Variable(Y[:m])

x_test = Variable(X[m:])
y_test = Variable(Y[m:])

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__() #call parent class initializer
        self.h1 = torch.nn.Linear(768, 100)
        self.h2 = torch.nn.Linear(100, 20)
        self.h3 = torch.nn.Linear(20, 4)
        self.out = torch.nn.Linear(4, 1) #hidden layer to single output

    #define the forward propagation/prediction equation of our model
    def forward(self, x):
        h1 = self.h1(x) #linear combination
        h1 = F.selu(h1) #activation
        h2 = self.h2(h1)
        h2 = F.selu(h2)
        h3 = self.h3(h2)
        h3 = F.selu(h3)
        out = self.out(h3) #linear combination
        out = F.sigmoid(out) #activation
        return out

#training hyper-parameters
no_epochs = 100
alpha = 0.00001 #learning rate

mynet = Net() #create model from class
criterion = torch.nn.MSELoss() #define cost criterion
optimizer = torch.optim.Rprop(mynet.parameters(), lr=alpha) #choose optimizer

#define graph for plotting costs
costs = [] #to store our calculated costs
plt.ion() 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_xlim(0, no_epochs)
plt.show()

#training loop
for epoch in range(no_epochs):

    #forward propagate - calulate our hypothesis
    h = mynet.forward(x_train)

    #calculate, plot and print cost
    cost = criterion(h, y_train)
    costs.append(cost.data[0])
    ax.plot(costs, 'b')
    fig.canvas.draw()    
    print('Epoch ', epoch, ' Cost: ', cost.data[0])

    #backpropagate + gradient descent step
    optimizer.zero_grad() #set gradients to zero
    cost.backward() #backpropagate to calculate derivatives
    optimizer.step() #update our weights
    plt.pause(0.001)

#test accuracy
test_h = mynet.forward(x_test) #predict values for out test set
test_h.data.round_() #round output probabilities to give us discrete predictions
correct = test_h.data.eq(y_test.data) #perform element-wise equality operation
accuracy = torch.sum(correct)/correct.shape[0] #calculate accuracy
print('Test accuracy: ', accuracy)

torch.save(mynet.state_dict(), 'mynet_trained.pt') #save our model parameters
mynet.load_state_dict(torch.load('mynet_trained.pt')) #load in model parameters