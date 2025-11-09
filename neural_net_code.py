import torch
torch.manual_seed(123)
import torch.nn as nn
import torch.nn.functional as F
class MyGameDecider(nn.Module):
    def __init__(self):
        super(MyGameDecider, self).__init__()
        # Initialize the modules we need to build the network
        self.conv=nn.Conv2d(in_channels=4,out_channels=16,kernel_size=3, stride=1)
        self.fc=nn.Linear(16*16*18+1,8)

    def forward(self,my_position,their_poistion,my_trail,their_trail,boost_count):
        # Perform the calculation of the model to determine the prediction
        image_tensor=torch.cat(
            [my_position.unsqueeze(1), 
             their_poistion.unsqueeze(1), 
             my_trail.unsqueeze(1), 
             their_trail.unsqueeze(1)], 
            dim=1
        )
        maps=self.conv(image_tensor)
        maps=F.relu(maps)
        flat=maps.view(-1,16*16*18)
        if boost_count.dim() == 1:
            boost_count = boost_count.unsqueeze(1)
        merged_features=torch.cat((flat,boost_count.float()),dim=1)
        logic=self.fc(merged_features)
        x=F.log_softmax(logic,dim=1)
        return x
model=MyGameDecider()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
probabilities=[]
game_result=0
def calculate_policy_loss(policy_log_probs, cumulative_reward):
    reward_tensor = torch.tensor(cumulative_reward, dtype=torch.float32)
    loss = -torch.mean(policy_log_probs * reward_tensor)   
    return loss
def train_model():
    global probabilities, game_result, model, optimizer
    model.train()
    loss = calculate_policy_loss(move_probalities,game_result)
    ## Step 4: Perform backpropagation
    # Before calculating the gradients, we need to ensure that they are all zero.
    # The gradients would not be overwritten, but actually added to the existing ones.
    optimizer.zero_grad()
    # Perform backpropagation
    loss.backward()
    ## Step 5: Update the parameters
    optimizer.step()
def play_game(model):
    if end_game==True:
        if loss:
            game_result=torch.tensor(1)
        elif win:
            game_result=torch.tensor(-1)
        else:
            game_result=torch.tensor(.1)
        train_model(model,optimizer,probabilities,game_result)
        probabilities=[]
        game_result=0
    log_probs_all = model(my_position, their_poistion, my_trail, their_trail, boost_count)
    dist = Categorical(logits=log_probs_all.squeeze(0))
    action_index_tensor = dist.sample()
    log_prob_of_chosen_action = dist.log_prob(action_index_tensor)
    probabilities.append(log_prob_of_chosen_action)