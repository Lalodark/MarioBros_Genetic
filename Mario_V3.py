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

from keras.models import save_model
from keras.models import load_model

from PIL import Image

SIMPLE_MOVEMENT= [['NOOP'],['right'],['right', 'A'],['right', 'B'],['right', 'A', 'B'],['A'],['left']]

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

action_space = env.action_space.n
#state_size = (80, 88, 1)
state_size = (10, 20, 1)
#state_size = (20, 30, 1)

total_reward = 0
done = True

def preprocess_state(state):
    image = Image.fromarray(state)
    image = image.resize((22, 20))
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
    weights_1 = Mario(state_size,action_space,self).main_network.get_weights()
    weights_2 = Mario(state_size,action_space,partner).main_network.get_weights()

    new_weights = weights_1

    a = random.randint(0,len(new_weights)-1)
    b = random.randint(0,len(new_weights[a])-1)
    
    try:
        c = random.randint(0,len(new_weights[a][b])-1)
    except:
        c = -1
        

    if c > -1:
        new_weights[a][b][c] = weights_2[a][b][c]
    else:
        new_weights[a][b] = weights_2[a][b]

    child = Mario(state_size,action_space)
    child.main_network.set_weights(new_weights)
    newgenes = DNA(child)
    return newgenes
    

  """ def mutate(self, m, child):
    for i in range(len(child.genes)):
      if type(child.genes[i][0]) == type([]) or type(child.genes[i][0]) == type(np.array(8)):
        for j in range(len(child.genes[i])):
          if type(child.genes[i][j][0]) == type([]) or type(child.genes[i][j][0]) == type(np.array(8)):
            for k in range(len(child.genes[i][j])):
              if type(child.genes[i][j][k][0]) == type([]) or type(child.genes[i][j][k][0]) == type(np.array(8)):
                for l in range(len(child.genes[i][j][k])):
                  for n in range(len(child.genes[i][j][k][l])):
                    if child.genes[i][j][k][l][n] != 0:
                      if (random.random() < n): child.genes[i][j][k][l][n] += random.uniform(-0.5, 0.5)
              else: 
                for l in range(len(child.genes[i][j][k])):
                  if child.genes[i][j][k][l] != 0:
                    if (random.random() < m): child.genes[i][j][k][l] += random.uniform(-0.5, 0.5)
          else:
            for k in range(len(child.genes[i][j])):
              if child.genes[i][j][k] != 0:
                if (random.random() < m): child.genes[i][j][k] += random.uniform(-0.5, 0.5)
      else:
        for j in range(len(child.genes[i])):
          if child.genes[i][j] != 0:
            if (random.random() < m): child.genes[i][j] += random.uniform(-0.5, 0.5)
    return child """

  def mutate(self, m, child):
    new_weights = Mario(state_size,action_space,child).main_network.get_weights()      
    for i in range(0, 2000):         
      r = random.random()         
      if r < m:             
        a = random.randint(0,len(new_weights)-1)             
        b = random.randint(0,len(new_weights[a])-1)                          
        try:                 
          c = random.randint(0,len(new_weights[a][b])-1)             
        except:                 
          c = -1                               
        if c > -1:                 
          new_weights[a][b][c] += random.uniform(-0.5, 0.5)             
        else:                 
          new_weights[a][b] += random.uniform(-0.5, 0.5)     
    
    hola = Mario(state_size,action_space)
    hola.main_network.set_weights(new_weights)
    
    
    return DNA(hola)


class Population:
  def __init__(self, m, num):
    self.mutationRate = m
    self.population = [0]*num
    self.generations = 0
    for i in range(num):
      self.population[i] = Mario(state_size,action_space)

  def calcMaxFitness(self, popu):
    maxFitness = 0
    secondMaxFitness = 0
    mario = 0
    second_mario = 0
    for i in range(len(popu.population)):
      current_fitness = popu.population[i].fitness
      if (current_fitness > maxFitness):
        secondMaxFitness = maxFitness
        second_mario = mario
        maxFitness = current_fitness
        mario = i
      else:  
        if current_fitness > secondMaxFitness and current_fitness < maxFitness:
          secondMaxFitness = current_fitness
          second_mario = i
    return [maxFitness, mario, second_mario]

  def getMario(self, index):
    return self.population[index]

  def acceptReject(self, maxFitness, popu):
    besafe = 0
    while(True):
      index = random.randint(0,len(popu.population)-1)
      partner = DNA(popu.population[index])
      r = random.randint(round(maxFitness/4),round(maxFitness))
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
    maxFitness, best_mario, second_mario = popu.calcMaxFitness(popu)
    partnerA = DNA(popu.population[best_mario])
    partnerB = partnerA
    for i in range(len(popu.population)):
      maxFitness, best_mario, second_mario = popu.calcMaxFitness(popu)
      if i != best_mario:
        r = random.randint(0, 1)

        #if r == 0:
        partnerA = DNA(popu.population[best_mario])
        partnerB = partnerA
        #    partnerB = DNA(popu.population[second_mario])
        #else:
        #    partnerB = DNA(popu.population[best_mario])
        #    partnerA = DNA(popu.population[second_mario])


        if i == 0:
            print("Padres: {} y {}".format(str(partnerA.fitness), str(partnerB.fitness)))

        #child_dna = partnerA.crossover(partnerB)
        child_dna = partnerA
        child_dna = child_dna.mutate(popu.mutationRate, child_dna)
        newPopulation[i] = Mario(state_size,action_space,child_dna)
      else:
        newPopulation[i] = popu.population[best_mario]


    
    popu.population = newPopulation
    popu.generations+=1
    return popu

  def getGenerations(self):
    return self.generations


class Mario:        
    def __init__(self, state_size, action_size, dna = None):
      self.state_space = (70,)
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
        model.add(Dense(70, input_shape=self.state_space, activation='relu'))

        model.add(Dense(self.action_space, input_shape=self.state_space, activation='linear'))
        
        model.compile(loss='mse', optimizer='adam') #, optimizer=keras.optimizers.RMSprop(0.01)
        return model

    def calc_fitness(self, distance, y_pos):
      self.fitness = math.floor((distance/100)**3 + y_pos*0.1)
    
    def act(self, state, acting):
      if acting:
        state_array = np.array(state.ravel(),dtype=float)
        for i in range(len(state_array)):
          state_array[i] = float(state_array[i]/255)
          
        neural_input = np.atleast_2d(state_array)
        Q_value = self.main_network.predict_step(neural_input)[0]
        self.chosenAction = np.argmax(Q_value)
        return self.chosenAction
      else:
        return self.chosenAction


def juego(popu,num_timesteps,onGround,j):
    state = preprocess_state(env.reset())
    state = state.reshape(-1,20, 22, 1)
    time_step = 0
    total_distance = 0
    dqn = popu.population[j]
    max_time=0
    distance = 0
    onGround = 79
    x_pos = 0
    stage = 0
    world = 0
    acting = True
    if i == 0:
      dqn.main_network = load_model("hola.h5")
    for t in range(num_timesteps):
        time_step +=1
        env.render()
        #action = dqn.act(state[0:1,60:70,30:50,0:1], onGround)
        action = dqn.act(state[0:1,10:15,0:14,0:1], acting)
        #action = dqn.act(state[0:1,3:19,3:22,0:1], acting)
        next_state, reward, done, info = env.step(action)
        #plt.plot(next_state[0])
        next_state = preprocess_state(next_state)
        state = next_state
        state = state.reshape(-1,20, 22, 1)
        
        if onGround < info['y_pos']:
          acting = False
        else:
          acting = True
        
        onGround = info['y_pos']

        if info['x_pos'] > x_pos or info['stage'] > stage or info['world'] > world:
          x_pos = info['x_pos']
          stage = info['stage']
          world = info['world']
          distance = info['x_pos'] + 10000 * (info['stage']-1) + 100000 * (info['world']-1)
          max_time = time_step
        
        if done or max_time<time_step-1000 or info['life'] < 2:
            break
    dqn.calc_fitness(distance, onGround)

    if dqn.fitness == 79:
      dqn.fitness = 0
    
    return dqn


mutation_rate = 0.01
population_size = 20

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

    #if i == 0:
    #  mario = Mario(state_size,action_space)
    #  mario.main_network = load_model("hola.h5")
    #  child_dna = DNA(mario)
    #  for j in range(population_size):
    #    child_dna = child_dna.mutate(popu.mutationRate, child_dna)
    #    popu.population[j] = Mario(state_size,action_space, child_dna.genes)

    print("Generation is {}".format(str(i)))
    for j in range(population_size):
        dqn = juego(popu,num_timesteps,onGround,j)
        print("Mario actual: {} con fitness: {}".format(str(j+1), str(dqn.fitness)))


    maxFitness, mario, second_mario = popu.calcMaxFitness(popu)
    best_mario = popu.getMario(mario)
    other_mario = popu.getMario(second_mario)
    print("---------------------------------------------------")
    print("Gen's {} best Mario is {}, with fitness = {}".format(str(i), str(mario), str(maxFitness)))
    #if i % 1:
    best_mario.main_network.save("hola.h5")
    other_mario.main_network.save("otro_mario.h5")

    #if i % 5:
    #  mutation_rate = mutation_rate * 0.6
    
    popu = popu.reproduction(popu)
    clear_output(wait = True)
env.close()