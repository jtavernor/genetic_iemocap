import logging

import torch
import random
import numpy as np

import neat_pytorch_linear.population as pop
import genetic_iemocap.training.pytorch_neat.evolved_feedforward_config_small as c
from neat_pytorch_linear.visualize import draw_net
from neat_pytorch_linear.phenotype.feed_forward import FeedForwardNet
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

logger = logging.getLogger(__name__)

logger.info(c.IEMOCAPConfig.DEVICE)
neat_instance = pop.Population(c.IEMOCAPConfig)
solution, generation = neat_instance.run()

print('avg_fitness', neat_instance.Config.avg_fitness)
print('avg_size', neat_instance.Config.avg_generation_num_params)
plt.plot(neat_instance.Config.avg_fitness)
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.savefig('average_fitnesses_small.png')

plt.clf()

plt.plot(neat_instance.Config.avg_generation_num_params)
plt.xlabel('Generation')
plt.ylabel('Average Number of Params')
plt.savefig('average_params_small.png')

if solution is not None:
    logger.info('Found a Solution in generation', generation)

    test_accuracy = neat_instance.Config.get_test_accuracy(solution)

    logger.info('Solution accuracy:', test_accuracy)

    draw_net(solution, view=True, filename='./images/feedforward_small_solution', show_disabled=True)
else:
    population = neat_instance.population
    fitnesses = [max(0, neat_instance.Config.fitness_fn(genome, gen_check=False)) for genome in population]
    best_fitness_idx = np.argmax(fitnesses)
    best_individual = population[best_fitness_idx]

    test_accuracy, num_params = neat_instance.Config.get_test_accuracy(best_individual)

    logger.info('Best individual in final population:', test_accuracy)
    print('Best individual in final population:', test_accuracy)
    print('model size', num_params)

    draw_net(best_individual, view=True, filename='./images/feedforward_small_best_ind', show_disabled=True)
