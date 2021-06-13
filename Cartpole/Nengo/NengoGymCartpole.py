import nengo
import numpy as np
import gym
from scipy.special import softmax

TIMESTEP = 0.001

## Nengo-Gym Interface object

## Object holds attributes such as the reward, state, and action of the Gym environment

## It holds a counter, used in case we want to de-synchronize Nengo simulation updates with Gym environment updates

## It shapes the reward (and sends to the SNN the shaped version). This is due to the fact that the rewards given by CartPole Gym environment are always 1, which is inadequate in our setup.

## It has 3 class methods it uses to interface with a Nengo simulation
class NengoGymCartpole(object):
    def __init__(self,update_each=1):
        print("Gym Init")        
        self.feedback = []
        self.controls = []
        
        self.env = gym.make("CartPole-v0")
        
        print("Action Space:")
        print(self.env.action_space)
     
        
        print("Observation Space:")
        print(self.env.observation_space)
        print(self.env.observation_space.high)
        print(self.env.observation_space.low)
        
        self.shaped_reward = 0
        self.true_reward = 0
        self.total_reward = 0
        self.steps = 0
        
        self.counter = 0

        self.state = self.env.reset()
    
        self.update_each = update_each
        
        self.true_total_rewards = []
    
    def get_shaped_reward(self):
    #   Observation               Min                     Max
    #   Cart Position             -4.8                    4.8
    #   Cart Velocity             -Inf                    Inf
    #   Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    #   Pole Angular Velocity     -Inf                    Inf

        x = self.state
    
        cart_position = np.abs(x[0])
        cart_velocity = np.abs(x[1])
        cart_mid = 1.2
        
        cart_reward = (cart_mid - cart_position) / cart_mid
    
        pole_angle = np.abs(x[2])
        pole_velocity = np.abs(x[3])
        pole_mid = 0.104
        
        pole_velocity_reward = - 0.5 * np.sign(x[2]) * np.sign(x[3])
            
        pole_reward = (pole_mid - pole_angle) / pole_mid
        
        reward = pole_reward
        
#         reward = 50 * np.cos(pole_angle)
        
        return reward
    
    def get_state(self, t):
        return self.state
    
    def get_reward(self, t):
        return self.shaped_reward
    
    def set_action(self, t, controls):
        self.counter = self.counter + 1
        
        if self.counter == self.update_each :
            self.counter = 0
#             print('ACTION TAKEN', self.counter)
        else:
#             print('NO ACTION TAKEN', self.counter)
            return
            
        
        softmaxed = softmax(controls)
        sampled = np.random.choice([0, 1], size=1, p=softmaxed)[0]
        argmaxed = np.argmax(softmaxed)
        action = argmaxed
    
        prev_state = self.env.state
        
        self.state, self.true_reward, done, info = self.env.step(action) 
        self.shaped_reward = self.get_shaped_reward()

        self.env.render() #one frame
        
        self.total_reward += self.true_reward
        
        self.steps += 1
        
        if done:
            self.true_total_rewards.append(self.total_reward)
               
            self.steps = 0
            self.state = self.env.reset()
            self.total_reward = 0