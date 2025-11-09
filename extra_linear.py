import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
player_name=0
from case_closed_game import Game, Direction, GameResult
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
game_count=0
probabilities=[]
my_trail=[]
their_trail=[]
boost_count=0
turn=0
reward=[]
class MyGame(nn.Module):
    def __init__(self):
        super(MyGame, self).__init__()
        # Initialize the modules we need to build the network
        self.conv=nn.Conv2d(in_channels=4,out_channels=16,kernel_size=3, stride=1)
        self.conv1=nn.Conv2d(in_channels=16,out_channels=64,kernel_size=4, stride=1)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=256,kernel_size=6, stride=3)
        self.fc1=nn.Linear(3074,30)
        self.fc2=nn.Linear(30,20)
        self.fc3=nn.Linear(20,8)

    def forward(self,my_position,their_poistion,my_trail,their_trail,boost_count,name):
        # Perform the calculation of the model to determine the prediction
        image_tensor=torch.cat(
            [my_position.unsqueeze(1), 
             their_poistion.unsqueeze(1), 
             my_trail.unsqueeze(1), 
             their_trail.unsqueeze(1)], 
            dim=1
        )
        image_tensor = image_tensor.permute(1, 0, 2).unsqueeze(0)
        maps=self.conv(image_tensor)
        maps=F.relu(maps)
        maps=self.conv1(maps)
        maps=F.relu(maps)
        maps=self.conv2(maps)
        maps=F.relu(maps)
        flat=maps.view(-1,3072)
        boost_count = boost_count.unsqueeze(0)
        boost_count = boost_count.unsqueeze(0)
        name=torch.tensor(name)
        name=name.unsqueeze(0)
        name=name.unsqueeze(0)
        merged_features=torch.cat((flat,boost_count,name),dim=1)
        x=self.fc1(merged_features)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        logic=self.fc3(x)
        x=F.log_softmax(logic,dim=1)
        return x

model=MyGame()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
game_result=0
#Edit out next three lines when training new model

checkpoint = torch.load('extra_linear_model_state.txt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def calculate_policy_loss(policy_log_probs, cumulative_reward):
    loss=torch.tensor(0)
    for i in policy_log_probs:
        loss=loss+i*cumulative_reward
    loss=-1*loss/len(policy_log_probs)
    reward.append(-loss.item())
    if len(reward)>20:
        del reward[0]
    return loss
def train_model():
    global probabilities, game_result, model, optimizer, game_count
    model.train()
    loss = calculate_policy_loss(probabilities,game_result)
    ## Step 4: Perform backpropagation
    # Before calculating the gradients, we need to ensure that they are all zero.
    # The gradients would not be overwritten, but actually added to the existing ones.
    optimizer.zero_grad()
    # Perform backpropagation
    loss.backward()
    ## Step 5: Update the parameters
    optimizer.step()
    game_count+=1
    state = {
        'epoch': game_count,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Optionally save the total number of games played or other metrics
    }
    if game_count%20==0:
        with open("linear_data.txt", "a") as file:
            file.write(str(sum(reward)/len(reward))+'\n')
        torch.save(state, "./extra_linear_model_state.txt")
# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    global my_trail,their_trail,boost_count
    if player_name==1:
        my_trail=data["agent1_trail"]
        boost_count=data['agent1_boosts']
        their_trail=data['agent2_trail']
    else:
        my_trail=data['agent2_trail']
        boost_count=data['agent2_boosts']
        their_trail=data["agent1_trail"]
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    global player_name, turn
    turn=0
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    player_name=data["player_number"]
    print("player"+str(player_name))
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    global probabilities,turn
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    # Simple example: always go RIGHT (replace this with your logic)
    # To use a boost: move = "RIGHT:BOOST"
    my_p=torch.zeros(20, 18)
    their_p=torch.zeros(20,18)
    my_t=torch.zeros(20,18)
    their_t=torch.zeros(20,18)
    boost_c=torch.tensor(boost_count)
    for i in range(len(my_trail)-1):
        my_x=my_trail[i][0]
        my_y=my_trail[i][1]
        my_t[my_x][my_y]=1
    for i in range(len(their_trail)-1):
        my_x=their_trail[i][0]
        my_y=their_trail[i][1]
        their_t[my_x][my_y]=1
    if len(their_trail)>0:
        my_x=their_trail[-1][0]
        my_y=their_trail[-1][1]
        their_p[my_x][my_y]=1
    if len(my_trail)>0:
        my_x=my_trail[-1][0]
        my_y=my_trail[-1][1]
        my_p[my_x][my_y]=1
    val=-1
    if player_name==2:
        val=1
    log_probs_all = model(my_p, their_p, my_t, their_t, boost_c,val)
    dist = Categorical(logits=log_probs_all.squeeze(0))
    #action_index_tensor = dist.sample()
    action_index_tensor=torch.argmax(log_probs_all.squeeze(0))
    log_prob_of_chosen_action = dist.log_prob(action_index_tensor)
    log_prob_of_chosen_action=log_prob_of_chosen_action.unsqueeze(0)
    probabilities.append(log_prob_of_chosen_action.squeeze(0))
    move = action_index_tensor
    if turn<4:
        move=torch.randint(0,8,size=(1,1))
    turn+=1
    match move:
        case 0:
            move='UP'
        case 1:
            move='DOWN'
        case 2:
            move='LEFT'
        case 3:
            move='RIGHT'
        case 4:
            move="UP:BOOST"
        case 5:
            move="DOWN:BOOST"
        case 6:
            move="LEFT:BOOST"
        case 7:
            move="RIGHT:BOOST"
    # Example: Use boost if available and it's late in the game
    # turn_count = state.get("turn_count", 0)
    # if boosts_remaining > 0 and turn_count > 50:
    #     move = "RIGHT:BOOST"
    # -----------------end code here--------------------
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    global model,optimizer,probabilities,game_result, player_name
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    status=data['result']
    s=3
    if(status=='AGENT1_WIN'):
        s=1
    elif(status=='AGENT2_WIN'):
        s=2
    else:
        s=3
    print("Results:"+str(s))
    print(player_name)
    status=s
    if status==3:
        gameame_result=torch.tensor(.1)
        print("tie")
    elif player_name==status:
        game_result=torch.tensor(-1)
        print("win")
    else:
        game_result=torch.tensor(1)
        print("lost")
    #train_model()
    print(player_name)
    probabilities=[]
    game_result=0
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)
