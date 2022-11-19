import tensorflow as tf
from tensorflow import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from IPython.display import clear_output
import math
from math import floor
#from matplotlib import pyplot as plt
import gc

from keras.models import save_model
from keras.models import load_model

from PIL import Image

SIMPLE_MOVEMENT= [['NOOP'],['right'],['right', 'A'],['A'],['left']]

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

action_space = env.action_space.n
#state_size = (80, 88, 1)
#state_size = (10, 20, 1)
state_size = (20, 30, 1)

total_reward = 0
done = True

def preprocess_state(state):
    image = Image.fromarray(state)
    image = image.resize((44, 40))
    image = image.convert('L')
    image = np.array(image)

    return image

env.reset()

class DNA:
  def __init__(self, mario): 
    self.genes = mario.main_network.get_weights()
    self.fitness = mario.fitness

    #def calcFitness(self, x_pos, y_pos, life):
      #self.fitness = x_pos + y_pos + life*100
  
  def crossover(self, partner):
    parents = partner.genes
    child = Mario(state_size,action_space)
    for i in range(len(partner.genes)):
      if type(partner.genes[i][0]) == type([]) or type(partner.genes[i][0]) == type(np.array(8)):
        for j in range(len(partner.genes[i])):
          if type(partner.genes[i][j][0]) == type([]) or type(partner.genes[i][j][0]) == type(np.array(8)):
            for k in range(len(partner.genes[i][j])):
              if type(partner.genes[i][j][k][0]) == type([]) or type(partner.genes[i][j][k][0]) == type(np.array(8)):
                for l in range(len(partner.genes[i][j][k])):
                  crossover = random.randint(0,len(partner.genes[i][j][k][l]))
                  for m in range(len(partner.genes[i][j][k][l])):
                    if (j > crossover): parents[i][j][k][l][m] = self.genes[i][j][k][l][m]
              else:
                crossover = random.randint(0,len(partner.genes[i][j][k]))
                for l in range(len(partner.genes[i][j][k])):
                  if (l > crossover): parents[i][j][k][l] = self.genes[i][j][k][l]
          else:
            crossover = random.randint(0,len(partner.genes[i][j]))
            for k in range(len(partner.genes[i][j])):
              if (k > crossover): parents[i][j][k] = self.genes[i][j][k]
      else:
        crossover = random.randint(0,len(partner.genes[i]))
        for j in range(len(partner.genes[i])):
          if (j > crossover): parents[i][j] = self.genes[i][j]
    
    child.main_network.set_weights(parents)
    newgenes = DNA(child)
    return newgenes

  def mutate(self, m, child):
    for i in range(len(child.genes)):
      if type(child.genes[i][0]) == type([]) or type(child.genes[i][0]) == type(np.array(8)):
        for j in range(len(child.genes[i])):
          if type(child.genes[i][j][0]) == type([]) or type(child.genes[i][j][0]) == type(np.array(8)):
            for k in range(len(child.genes[i][j])):
              if type(child.genes[i][j][k][0]) == type([]) or type(child.genes[i][j][k][0]) == type(np.array(8)):
                for l in range(len(child.genes[i][j][k])):
                  for n in range(len(child.genes[i][j][k][l])):
                    if child.genes[i][j][k][l][n] != 0:
                      if (random.random() < n): child.genes[i][j][k][l][n] = random.uniform(-0.5, -0.5)
              else: 
                for l in range(len(child.genes[i][j][k])):
                  if child.genes[i][j][k][l] != 0:
                    if (random.random() < m): child.genes[i][j][k][l] = random.uniform(-0.5, -0.5)
          else:
            for k in range(len(child.genes[i][j])):
              if child.genes[i][j][k] != 0:
                if (random.random() < m): child.genes[i][j][k] = random.uniform(-0.5, -0.5)
      else:
        for j in range(len(child.genes[i])):
          if child.genes[i][j] != 0:
            if (random.random() < m): child.genes[i][j] = random.uniform(-0.5, -0.5)
    return child



class Population:
  def __init__(self, m, num):
    self.mutationRate = m
    self.population = [0]*num
    self.generations = 0
    for i in range(num):
      self.population[i] = Mario(state_size,action_space)

  def calcMaxFitness(self, popu):
    maxFitness = 40
    secondMaxFitness = 0
    mario = 0
    second_mario = 0
    for i in range(len(popu.population)):
      if (popu.population[i].fitness > maxFitness):
        maxFitness = popu.population[i].fitness
        mario = i
      if popu.population[i].fitness > secondMaxFitness and popu.population[i].fitness < maxFitness:
        secondMaxFitness= popu.population[i].fitness
    return [maxFitness, mario, second_mario]

  def getMario(self, index):
    return self.population[index]

  def acceptReject(self, maxFitness, popu):
    besafe = 0
    while(True):
      index = random.randint(0,len(popu.population)-1)
      partner = DNA(popu.population[index])
      r = random.randint(0,round(maxFitness))
      if (r <= partner.fitness):
        return partner
      besafe+=1

      if (besafe > 10000):
        return None



    
  def reproduction(self, popu):
    maxFitness = 0
    for i in range(len(popu.population)):
      if (popu.population[i].fitness > maxFitness):
        maxFitness = popu.population[i].fitness

    newPopulation = [0]*len(popu.population)
    for i in range(len(popu.population)):
      partnerA = popu.acceptReject(maxFitness, popu)
      partnerB = popu.acceptReject(maxFitness, popu)
      print("Mario Padres: {} y {}".format(str(partnerA.fitness), str(partnerB.fitness)))
      child = partnerA.crossover(partnerB)
      child = child.mutate(popu.mutationRate, child)
      newPopulation[i] = Mario(state_size,action_space,child)
    
    popu.population = newPopulation
    popu.generations+=1
    return popu

  def getGenerations(self):
    return self.generations


class Mario:        
    def __init__(self, state_size, action_size, dna = None):
      self.state_space = (200,)
      self.action_space = action_size
      self.memory = deque(maxlen=5000)
      self.chosenAction = 0
      self.distance = 0
      self.fitness = 0
      
      self.main_network = self.build_network()
      if dna is not None:
        self.main_network.set_weights(dna.genes)

    def build_network(self):
        model = Sequential()
        model.add(Dense(200, input_shape=self.state_space, activation='relu'))
        
        model.add(Dense(512, input_shape=self.state_space, activation='relu'))

        model.add(Dense(64, input_shape=self.state_space, activation='relu'))
        
        model.add(Dense(16, input_shape=self.state_space, activation='relu'))
        
        #model.add(Dense(1, activation='relu'))
        model.add(Dense(1, input_shape=self.state_space, activation='linear'))
        
        model.compile(loss='mse', optimizer='adam') #, optimizer=keras.optimizers.RMSprop(0.01)
        return model

    def calc_fitness(self, distance, y_pos):
      self.fitness = (distance/100)**2 + y_pos
    
    def act(self, state, acting):
      if acting:
        neural_input = np.atleast_2d(state.ravel())
        Q_value = self.main_network.predict_step(neural_input)
        self.chosenAction = floor((Q_value[0][0]*100) % self.action_space)
        return self.chosenAction
      else:
        return self.chosenAction



def juego(popu,num_timesteps,onGround,j):
    state = preprocess_state(env.reset())
    state = state.reshape(-1,40, 44, 1)
    time_step = 0
    dqn = popu.population[j]
    max_time=0
    distance = 0
    onGround = 100
    acting = True

    # if i == 0:
    #   dqn.main_network = load_model("hola.h5")
    
    for t in range(num_timesteps):
        time_step +=1

        env.render()
        #action = dqn.act(state[0:1,60:70,30:50,0:1], onGround)
        action = dqn.act(state[0:1,20:30,10:30,0:1], acting)
        next_state, reward, done, info = env.step(action)
        #plt.plot(next_state[0])
        
        next_state = preprocess_state(next_state)
        state = next_state
        state = state.reshape(-1,40, 44, 1)
        
        if onGround < info['y_pos']:
          acting = False
        else:
          acting = True
        
        onGround = info['y_pos']

        if info['x_pos'] > distance:
          distance = info['x_pos'] + 10000 * (info['stage']-1) + 100000 * (info['world']-1)
          max_time = time_step
        
        if done or max_time<time_step-200 or info['life'] < 2:
            break
    dqn.calc_fitness(distance, onGround)
    return dqn


mutation_rate = 0.001
population_size = 50

num_episodes = 100000
num_timesteps = 500000
debug_length = 300

popu = Population(mutation_rate,population_size)


stuck_buffer = deque(maxlen=debug_length)


for i in range(num_episodes):
    Return = 0
    done = False
    onGround = 79

    state = preprocess_state(env.reset())

    print("Generation is {}".format(str(i)))
    for j in range(population_size):
        
        dqn = juego(popu,num_timesteps,onGround,j)
        print("Mario actual: {} con fitness: {}".format(str(j+1), str(dqn.fitness)))


    maxFitness, mario, second_mario = popu.calcMaxFitness(popu)
    best_mario = popu.getMario(mario)
    other_mario = popu.getMario(second_mario)
    print("---------------------------------------------------")
    print("Gen's {} best Mario is {}, with fitness = {}".format(str(i), str(mario), str(maxFitness)))
    if i % 10:
      best_mario.main_network.save("hola.h5")
      other_mario.main_network.save("otro_mario.h5")
    
    popu = popu.reproduction(popu)
    clear_output(wait = True)
    gc.collect()


env.close()