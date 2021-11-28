import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, in_size, hidd_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidd_size)
        self.linear2 = nn.Linear(hidd_size, out_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class qLearning:
    def __init__(self, model, lr, gamma):
        #Storing and passing the values from agent
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.exp_Val = nn.MSELoss()

    #A Pytorch tensor
    def train_step(self, state, action, reward, next_state, over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
       
        #Ability to check multiple sizes
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            over = (over, )

        #Predicted Q values with the current state it is in
        pred = self.model(state)
        target = pred.clone()
        for index in range(len(over)):
            Q_Val = reward[index]
            if not over[index]:
                #r + y * max(next_predicted Q value)
                #max value of the next prediction
                Q_Val = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action[index]).item()] = Q_Val
    
        self.optimizer.zero_grad()
        loss = self.exp_Val(target, pred)
        loss.backward()

        self.optimizer.step()



