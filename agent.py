import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, qLearning
from helper import plot

#Constant parameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
#The learning rate
LR = 0.001

class Agent:

    def __init__(self):
        self.num_games = 0
        #We use epsilon to demonstrate the randomness
        self.epsilon = 0
        #Set the initial discount ot .9 or anything less than 1
        self.gamma = 0.9 
        #manages the memory of and pops off if overloaded
        self.memory = deque(maxlen=MAX_MEMORY)
        #Creating an instance of the trainer (States, ,Actions)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = qLearning(self.model, lr=LR, gamma=self.gamma)

    # Here we are able to store the 11 values 
    def get_state(self, game):
        head = game.snake[0]
        #Points around the head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        #boolean values for the game directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # The Different dirrections of danger + 3
            # Danger/collision straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger/collision right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger/collision left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Bool values where 1 is true while rest is falseSnake Directions + 4
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Locates food + 4
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        #Sets bools to 0's and 1's
        return np.array(state, dtype=int)

    #Remember streoes the current state, reward, and calculates next state as well as game over state
    def remember(self, state, action, reward, next_state, over):
        #Appends to its memory to rember
        self.memory.append((state, action, reward, next_state, over))

    def long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            small_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            small_sample = self.memory

        states, actions, rewards, next_states, overs = zip(*small_sample)
        self.trainer.train_step(states, actions, rewards, next_states, overs)

    #Trains for only game step
    def short_memory(self, state, action, reward, next_state, over):
        self.trainer.train_step(state, action, reward, next_state, over)

    def get_action(self, state):
        # This is where the gent does random moves for the actions
        self.epsilon = 80 - self.num_games
        last_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            last_move[move] = 1
        else:
            stateZ = torch.tensor(state, dtype=torch.float)
            pred = self.model(stateZ)
            move = torch.argmax(pred).item()
            last_move[move] = 1

        return last_move


def train():
    #Saves array values of training scores and avg scores, these are used to plot later
    train_scores = []
    avg_scores = []

    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    #Here we want to play the game, this is done till the script is over
    while True:
        # This gets the old state
        old_state = agent.get_state(game)

        # We want to get the move based on the current state
        last_move = agent.get_action(old_state)

        # This will then perform the action/move and get the state and get reward
        reward, over, score = game.play_step(last_move)
        new_state = agent.get_state(game)

        # We use the training of the short memeory
        agent.short_memory(old_state, last_move, reward, new_state, over)

        #This will then rember the short memory given
        agent.remember(old_state, last_move, reward, new_state, over)

        #This then checks if the game is over 
        if over:
            # Here we are able to reset the game whcih is set up in the game.py file
            game.reset()
            #We can then itereate the number of games played +1
            agent.num_games += 1
            agent.long_memory()

            #Checks the highscore
            if score > record:
                record = score
                agent.model.save()

            #Displayes player information
            print('Game', agent.num_games, 'Score', score, 'Record:', record)

            train_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            avg_scores.append(mean_score)
            plot(train_scores, avg_scores)


if __name__ == '__main__':
    train()