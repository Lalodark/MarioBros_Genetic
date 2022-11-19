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
from keras.models import save_model
from keras.models import load_model
from PIL import Image


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

action_space = env.action_space.n
state_size = (20, 30, 1)

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

  def mutate(self, m, child):
    new_weights = Mario(state_size,action_space,child).main_network.get_weights()  
    mutated = False
    for i in range(0,50):
      r = random.randint(0, 100)       
      if r < m * 100:             
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

        mutated = True     
    
    hola = Mario(state_size,action_space)
    hola.main_network.set_weights(new_weights)
    
    return [DNA(hola), mutated]

class generation_stats:
  def __init__(self, size):
    self.mutated = ['-']*size
    self.parentA_index = [0]*size
    self.parentB_index = [0]*size
    self.parentA_fitness = [0]*size
    self.parentB_fitness = [0]*size

  def set_individual(self, i, mutated, parentA_index, parentB_index, parentA_fitness, parentB_fitness):
    if mutated == True:
      self.mutated[i] = 'y'  
    else: 
      self.mutated[i] = 'n'
    self.parentA_index[i] = parentA_index
    self.parentB_index[i] = parentB_index
    self.parentA_fitness[i] = parentA_fitness
    self.parentB_fitness[i] = parentB_fitness
    return 
  
  def get_individual(self, i):
    return [self.mutated[i], self.parentA_index[i], self.parentB_index[i], self.parentA_fitness[i], self.parentB_fitness[i]]


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
      r = random.randint(round(0),round(maxFitness))
      if (r <= partner.fitness):
        return [partner, index]
      besafe+=1

      if (besafe > 10000):
        return None
    
  def reproduction(self, popu):
    newPopulation = [0]*len(popu.population)
    maxFitness, best_mario, second_mario = popu.calcMaxFitness(popu)
    
    partnerA = DNA(popu.population[best_mario])
    partnerB = partnerA

    gen_stats = generation_stats(len(popu.population))

    for i in range(len(popu.population)):
      maxFitness, best_mario, second_mario = popu.calcMaxFitness(popu)
      if i != best_mario:
        if i != second_mario:
          partnerA, indexA = popu.acceptReject(maxFitness, popu)
          partnerB, indexB = popu.acceptReject(maxFitness, popu)

          print("{} - Parents: {} ({}) & {} ({})".format(str(i), str(partnerA.fitness), str(indexA), str(partnerB.fitness), str(indexB)))

          child_dna = partnerA.crossover(partnerB)


          child_dna, mutated = child_dna.mutate(popu.mutationRate, child_dna)

          gen_stats.set_individual(i, mutated, indexA, indexB, partnerA.fitness, partnerB.fitness)

          newPopulation[i] = Mario(state_size,action_space,child_dna)
        else:
          gen_stats.set_individual(i, "~", "~", "~", "~", "~")
          newPopulation[i] = popu.population[second_mario]
      else:
        gen_stats.set_individual(i, "~", "~", "~", "~", "~")
        newPopulation[i] = popu.population[best_mario]
      
    print("\n")

    popu.population = newPopulation
    popu.generations+=1

    return popu, gen_stats


class Mario:        
    def __init__(self, state_size, action_size, dna = None):
      self.state_space = (304,)
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
        
        model.add(Dense(32, input_shape=self.state_space, activation='relu'))
        model.add(Dense(self.action_space, input_shape=self.state_space, activation='linear'))
        
        model.compile(loss='mse', optimizer='adam')
        return model

    def calc_fitness(self, distance, y_pos):
      self.fitness = math.floor((distance/100)**3 + y_pos*2)

      if self.fitness == 158:
        self.fitness = 0
    
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


def juego(popu,num_timesteps,onGround,j,i):
    state = preprocess_state(env.reset())
    state = state.reshape(-1,20, 22, 1)
    time_step = 0
    individual = popu.population[j]
    max_time=0
    distance = 0
    onGround = 100
    acting = True
    x_pos = 0
    stage = 0
    world = 0    
    if i == 0:
      individual.main_network = load_model("Final_version.h5")

    for t in range(num_timesteps):
        time_step +=1

        env.render()
        action = individual.act(state[0:1,3:19,3:22,0:1], acting)
        next_state, reward, done, info = env.step(action)
        
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
        
        if info['y_pos'] < 79:
            onGround = 0
        
        if done or max_time<time_step-100 or info['life'] < 2:
            break
    
    individual.calc_fitness(distance, onGround)

    return individual


#Parameters
mutation_rate = 0.1
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

    if i == 0:
        maxFitness, best_index, second_best_index = popu.calcMaxFitness(popu)
        best = popu.getMario(best_index)
        second_best = popu.getMario(second_best_index)
        
    for j in range(population_size):
        individual = juego(popu,num_timesteps,onGround,j,i)

        if i != 0:
          mutated, parentA_index, parentB_index, parentA_fitness, parentB_fitness = gen_stats.get_individual(j)
          print("Last Bests: {} & {}   -   Parents: {} ({}) & {} ({})   -   Gen {} - Mario {} - Mutated {} - Fitness {} ".format(str(best.fitness), 
                str(second_best.fitness), str(parentA_fitness), str(parentA_index), str(parentB_fitness), str(parentB_index), str(i), str(j), str(mutated), str(individual.fitness)))
        else:
          print("Gen {} - Mario {} - Fitness {} ".format(str(i), str(j), str(individual.fitness)))

    maxFitness, best_index, second_best_index = popu.calcMaxFitness(popu)
    best = popu.getMario(best_index)
    second_best = popu.getMario(second_best_index)

    print("-------------------------------------------------------")
    print("-------------------------------------------------------")
    print("Gen's {} bests:  {} ({}) and {} ({})".format(str(i), str(maxFitness), str(best_index), str(second_best.fitness), str(second_best_index)))

    # best.main_network.save("Ultimate Mario.h5")
    # second_best.main_network.save("Second Ultimate Mario.h5")
    
    popu, gen_stats = popu.reproduction(popu)
    clear_output(wait = True)
env.close()