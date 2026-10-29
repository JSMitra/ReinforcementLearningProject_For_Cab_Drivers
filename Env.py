# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

max_actions = 15

# returns a randoms location between [1,m] for the Cab Agent to start with
def get_random_state():
    return random.randint(0,m-1)

# returns a random hour between [1,t] for the Cab Agent to start with
def get_random_time():
    return random.randint(0,t-1)

# returns a random day between [1,d] for the Cab Agent to start with
def get_random_day():
    return random.randint(0,d-1)

def convert_to_one_hot_encoding(v, max_v):
    enc_v = [int(i==v) for i in range(0,max_v)]
    return enc_v

def get_state_space():
    state_space = [i for i in range(0,m)]
    return state_space

def get_action_space():
     # (0,0): cab agent needs a 1 hour break
    action_space = [(0,0)]
    action_space.extend([a for a in permutations(range(0,m),2)])
    return action_space

def decode_one_hot_encoded(one_hot_encoded):
    return np.argmax(one_hot_encoded)

def get_state_values(state):
    m_decoded = decode_one_hot_encoded(state[0:m])
    t_decoded = decode_one_hot_encoded(state[m:m+t])
    d_decoded = decode_one_hot_encoded(state[m+t:m+t+d])
    
    return m_decoded, t_decoded, d_decoded

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = get_action_space()
        self.state_space = get_state_space()
        # randomly set the context for the cab agent to start with
        # inception begins: a random location, at a random hour, on a random day
        self.state_init = (get_random_state(),get_random_time(),get_random_day())

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        # convert m,t,d into one hot encoded arrays
        state_m = convert_to_one_hot_encoding(state[0], m)
        state_t = convert_to_one_hot_encoding(state[1], t)
        state_d = convert_to_one_hot_encoding(state[2], d)
        
        # pack the arrays into a consolidated state
        state_encod = np.hstack((state_m, state_t, state_d))
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
      
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        # safety check to contain the number of requests reaching our agent to 15
        if requests >15:
            requests =15

        # incase of no requests, the cab agent will have to take a break
        if requests == 0:
            #print('forced break due to no requests')
            return [0], [(0,0)]
        
        possible_actions_indeces = random.sample(range(0, (m-1)*m + 1), requests)
        actions = [self.action_space[i] for i in possible_actions_indeces]
        
        return possible_actions_indeces, actions


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        # our cab agent wants to take rest; so will incur a cost of C
        if action == (0,0):
            return -C
        
        # our cab is servicing a request
        pickup_loc = action[0]
        drop_loc = action[1]
        current_location = state[0]
        current_time = state[1]
        current_day = state[2]
        
        # the time the cab agent spend honoring the request
        time_of_ride = Time_matrix[pickup_loc][drop_loc][current_time][current_day]
        
        # this is the time cab agent spends to reach from current location to pickup location
        pick_up_time = Time_matrix[current_location][pickup_loc][current_time][current_day]
        
        # fuel cost incurred by the cab agent
        ride_cost_to_cab_agent = C*(pick_up_time+time_of_ride)
        
        # revenue for the trip
        ride_revenue = R*time_of_ride
        
        # profit = money earned - money spent
        reward = ride_revenue - ride_cost_to_cab_agent
        
        #print(time_of_ride,pick_up_time, ride_cost_to_cab_agent, ride_revenue, reward)
        
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        current_location = state[0]
        current_time = state[1]
        current_day = state[2]
        
        if action == (0,0):
            next_loc = current_location
            # 1 hour spent in rest
            time_of_ride = 1
            # no pickup time
            pick_up_time = 0
        else:
            pickup_loc = action[0]
            drop_loc = action[1]
            next_loc = drop_loc
            # the time the cab agent spend honoring the request
            time_of_ride = Time_matrix[pickup_loc][drop_loc][current_time][current_day]
            # this is the time cab agent spends to reach from current location to pickup location
            pick_up_time = Time_matrix[current_location][pickup_loc][current_time][current_day]
        
        total_time = time_of_ride + pick_up_time
        
        current_time = int(current_time + total_time)
        
        # day transition: cab agent driving at 12:00 am
        if current_time >= t:
            current_time = current_time-t
            current_day = int((current_day + 1)%d)
        
        # next state will be the next location, new time, new/same day
        next_state = (next_loc,current_time,current_day)
        
        # return next state and time spent by the driver for this trip 
        return next_state, total_time


    def reset(self):
        return self.action_space, self.state_space, self.state_init
