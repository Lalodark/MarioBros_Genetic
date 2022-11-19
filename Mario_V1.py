import tensorflow as tf
from tensorflow import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils.conv_utils import conv_kernel_idxs
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from IPython.display import clear_output
import math
from math import floor
from matplotlib import pyplot as plt

import threading


from keras.models import save_model
from keras.models import load_model

import time

from PIL import Image


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

action_space = env.action_space.n
state_size = (10, 20, 1)

total_reward = 0
done = True

def preprocess_state(state):
    image = Image.fromarray(state)
    image = image.resize((88, 80))
    image = image.convert('L')
    #image.show()
    image = np.array(image)

    return image

env.reset()

# for step in range(100000):
#     env.render()

#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     print(info)
#     total_reward += reward
    
#     clear_output(wait=True)
# env.close()

class DNA:
  def __init__(self, mario): 
    self.genes = mario.main_network.get_weights()
    self.fitness = mario.distance

  def calcFitness(self, x_pos):
    self.fitness = x_pos

  def crossover(self, partner):
    parents = [0]*len(self.genes)
    child = Mario(state_size,action_space)
    crossover = random.randint(0,len(self.genes))
    for i in range(len(self.genes)):
      if (i > crossover): parents[i] = self.genes[i]
      else: parents[i] = partner.genes[i]
    
    child.main_network.set_weights(parents)
    newgenes = DNA(child)
    return newgenes

  def mutate(self, m):
    for i in range(len(self.genes)):
        for j in range(len(self.genes[i])):
            if (random.random() < m):
                self.genes[i][j] = random.uniform(-2,2)

class Population:
  def __init__(self, m, num):
    self.mutationRate = m
    self.population = [0]*num
    self.generations = 0
    for i in range(num):
      self.population[i] = Mario(state_size,action_space)

#   def targetReached(self):
#     for i in range(self.population.length):
#       if (self.population[i].flag_get): 
#         return true
#     return false

#   def calcFitness(self, x_pos):
#     for i in range(self.population.length):
#       self.population[i].calcFitness(x_pos)
  
  def acceptReject(self, maxFitness):
    besafe = 0
    while(True):
      index = random.randint(0,len(self.population)-1)
      partner = DNA(self.population[index])
      r = random.randint(0,maxFitness)
      if (r < partner.fitness):
        return partner
      besafe+=1

      if (besafe > 10000):
        return None
    
    
  def reproduction(self):
    maxFitness = 0
    for i in range(len(self.population)):
      if (self.population[i].distance > maxFitness):
        maxFitness = self.population[i].distance

    newPopulation = [0]*len(self.population)
    for i in range(len(self.population)):
      partnerA = self.acceptReject(maxFitness)
      partnerB = self.acceptReject(maxFitness)
      child = partnerA.crossover(partnerB)
      child.mutate(self.mutationRate)
      newPopulation[i] = Mario(state_size,action_space,child)
    
    self.population = newPopulation
    self.generations+=1
  
  def getGenerations(self):
    return self.generations


class Mario:        
    def __init__(self, state_size, action_size, dna = None):
      #Crear variables para nuestro agente
      self.state_space = state_size
      self.action_space = action_size
      self.memory = deque(maxlen=5000)
      self.chosenAction = 0
      self.distance = 0
      
      self.epsilon = 0.5
      self.max_epsilon = 1
      self.min_epsilon = 0.01
      self.decay_epsilon = 0.0001

      #Creamos la red neuronal para nuestro agente
      self.main_network = self.build_network()
      if dna is not None:
          self.main_network.set_weights(dna.genes)

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(64, (4,4), strides=4, padding='same', input_shape=self.state_space))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (4,4), strides=2, padding='same'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=2))
        model.add(GlobalAveragePooling2D())

        model.add(Dense(64, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='relu'))

        model.compile(loss='mse', optimizer='adam') #, optimizer=keras.optimizers.RMSprop(0.01)
        return model

    def calc_distance(self, distance):
        self.distance = distance
    
    def act(self, state, onGround):
        if onGround < 83:
            Q_value = self.main_network.predict(state)
            self.chosenAction = floor((Q_value[0][0]*10) % self.action_space)
            return self.chosenAction
        else:
            return self.chosenAction

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_pred_act(self, state):
        Q_values = self.main_network.predict(state)
        return np.argmax(Q_values[0])


def juego(popu,num_timesteps,onGround,j):
    state = preprocess_state(env.reset())
    state = state.reshape(-1,80, 88, 1)
    time_step = 0
    dqn = popu.population[j]
    max_time=0
    distance = 0
    
    for t in range(num_timesteps):
        time_step +=1
        
        env.render()
        
        action = dqn.act(state[0:1,60:70,30:50,0:1], onGround)
        next_state, reward, done, info = env.step(action)
        onGround = info['y_pos']
        stuck_buffer.append({info['x_pos']})
        
        next_state = preprocess_state(next_state)
        state = next_state
        state = state.reshape(-1,80, 88, 1)

        if info['x_pos'] > distance:
            distance = info['x_pos']
            max_time = time_step

        print("Generation is {}\nTotal Time: {}\nMax Distance: {}\nCurrent Distance: {}\nCurrent Mario is: {}".format(str(i), str(time_step), str(distance), str(info['x_pos']), str(j)))

        clear_output(wait= True)

        if done or max_time<time_step-100:
            break
    dqn.calc_distance(distance)


mutation_rate = 0.01
population_size = 20

num_episodes = 100000
num_timesteps = 5000
debug_length = 300

popu = Population(mutation_rate,population_size)

stuck_buffer = deque(maxlen=debug_length)


for i in range(num_episodes):
    Return = 0
    done = False
    onGround = 79

    state = preprocess_state(env.reset())
    threads = [0]*population_size 
    for j in range(population_size):
        juego(popu,num_timesteps,onGround,j)
        # threads[j] = threading.Thread(target=juego, args=(popu,num_timesteps,onGround,j))
        # threads[j].start()
        
        # state = preprocess_state(env.reset())
        # state = state.reshape(80, 88, 1)
        # time_step = 0
        # dqn = popu.population[j]
        # max_time=0
        # distance = 0
        
        # for t in range(num_timesteps):
        #     env.render()
        #     time_step +=1

        #     action = dqn.act(state, onGround)

        #     next_state, reward, done, info = env.step(action)
        #     onGround = info['y_pos']
        #     stuck_buffer.append({info['x_pos']})

        #     if info['x_pos'] > distance:
        #         distance = info['x_pos']
        #         max_time = time_step

        #     print("Generation is {}\nTotal Time: {}\nMax Distance: {}\nCurrent Distance: {}\nCurrent Mario is: {}".format(str(i), str(time_step), str(distance), str(info['x_pos']), str(j)))

        #     clear_output(wait= True)

        #     if done or max_time<time_step-200:
        #         break
        # dqn.calc_distance(distance)
    # for j in range(population_size):
    #     threads[j].join()
    popu.reproduction()


    clear_output(wait = True)
env.close()