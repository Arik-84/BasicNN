import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#create model class that inherits the nn.module
class Model(nn.Module):
    
    def __init__(self, in_features = 4, h1=8,h2=9, out_features = 3):
        super().__init__() 
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
df = pd.read_csv(url)
df['variety'] = df['variety'].replace('Setosa',0.0)
df['variety'] = df['variety'].replace('Versicolor',1.0)
df['variety'] = df['variety'].replace('Virginica',2.0)


 
# Set x,y 
X = df.drop('variety', axis =1)
y = df['variety']

X = X.values
y = y.values

# Train test and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)   #80% train and 20% test

#Convert the X features into float tensors and y labels into long tensors (tensors are essentially multi-dimensionall arrays)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


#Set criterion of model to measue the error... how far our predicitons are from data
criterion = nn.CrossEntropyLoss()
#choose an optimzer... using Adam optimizer and set learning rate
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

#Train the model... how many epachs?(epach basically an interation through the network)
epachs = 100
losses = []

for i in range(epachs):  
    y_pred = model.forward(X_train)   # go 'forward' and get a prediction
    loss = criterion(y_pred, y_train) # measure the loss...the predicted value vs the training value
    losses.append(loss.detach().numpy())

    if i % 10 == 0:  #print every 10 epochs 
        print(f'Epoch: {i} loss: {loss}')


    #do backpropagation 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# graph to visualize the learning/how the 'losses' are improving through epachs(iterations)
plt.plot(range(epachs),losses)
plt.ylabel('loss/error')
plt.xlabel('epoch')

with torch.no_grad(): #turn off the backpropagation
    y_eval = model.forward(X_test) #X_test are features from out test set, y_eval will be predicitons
    loss = criterion(y_eval, y_test)
    print(loss)

correct  = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')  #what type of flower our netwrok thinks it is

        if y_val.argmax().item() == y_test[i]:
            correct += 1
    
print(f'We got {correct} correct')



