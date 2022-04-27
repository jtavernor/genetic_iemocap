# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os

import neat
import torch
import torch.nn.functional as F
import numpy as np
import random

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.recurrent_net import RecurrentNet

from genetic_iemocap.data_providers.iemocap import IEMOCAPDataset
from genetic_iemocap.training.utils import CustomCollation

import matplotlib.pyplot as plt

# Set seeds for reproducibility -- these seeds cause total extinction hahah! bad choice, but results empirically are constantly good and consistent, and it also feels a bit off to set these seeds anyway
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

class GenomeEvaluator:
    def __init__(self, batch_size):
        self.cur_gen = 1
        self.generation_fitnesses = []
        self.generation_size = []
        self.avg_fitness = []
        self.avg_size = []
        self.train_dataset = IEMOCAPDataset('../../data_providers/labels_dim10.pk', avg_audio_features=True, dim_size=10)
        self.subset_size = 50
        self.batch_size = batch_size
        self.max_words = 5
        self.train_dataset.truncate(self.max_words)
        self.num_workers = 2
        self.collater = CustomCollation(max_len=self.max_words)
        self.DEVICE = 'cpu'
    
    def get_individual_fitness(self, genome, config):
        network = RecurrentNet.create(genome, config, self.batch_size)
        self.train_dataset.train()
        self.train_dataset.use_subset(self.subset_size)
        dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collater.custom_collate_fn)
        correct = 0
        with torch.no_grad():
            for audio_features, input_lengths, labels in dataloader:
                audio_features = audio_features.to(self.DEVICE, dtype=torch.float)
                activation_labels = torch.tensor(labels['act']).to(self.DEVICE)
                
                true_batch_size = audio_features.size(0)
                audio_features = audio_features.reshape(true_batch_size, -1) # squash dimensions together
                # We need to pad to maximum batch size because this neat implementation uses matrix multiplication
                # which should make the implementation run much faster, even without GPU acceleration, but requires
                # batch size to be consistent which it is not usually
                padding_needed = self.batch_size - true_batch_size
                mask = self.batch_size - padding_needed
                if padding_needed:
                    padding_values = torch.zeros(padding_needed, audio_features.size(1))
                    audio_features = torch.vstack([audio_features, padding_values])
                outputs = network.activate(audio_features)
                outputs = outputs[:mask]
                outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                correct += (outputs == activation_labels).float().sum()

        accuracy = correct / len(self.train_dataset)
        accuracy = accuracy.cpu().item()
        genome.fitness = accuracy
        return accuracy, network

    def __call__(self, genomes, config, debug=None):
        print('Evaluating fitness for generation', self.cur_gen)
        if not isinstance(genomes, list):
            return self.get_individual_fitness(genomes, config)[0]
        for i, (_, genome) in enumerate(genomes):
            accuracy, network = self.get_individual_fitness(genome, config)

            # size = sum(p.numel() for p in network.parameters() if p.requires_grad)
            # Size is harder to get from this model as it's not a true PyTorch model 
            # the implementation instead uses torch tensors and operations instead of implementing a PyTorch model
            network_params = [network.input_to_output, network.output_to_output, network.output_responses, network.output_biases, network.outputs]
            if network.n_hidden > 0:
                network_params.extend([network.input_to_hidden, network.hidden_to_hidden, network.output_to_hidden, network.hidden_to_output, network.hidden_responses, network.hidden_biases])
            size = sum(p.numel() for p in network_params)

            print(f'Individual {i} fitness: {accuracy:.4f} size: {size} num_hidden: {network.n_hidden} n_internal_steps: {network.n_internal_steps}')
            self.generation_fitnesses.append(accuracy)
            self.generation_size.append(size)
        avg_gen_fit = np.mean(self.generation_fitnesses)
        avg_gen_size = np.mean(self.generation_size)
        self.avg_fitness.append(avg_gen_fit)
        self.avg_size.append(avg_gen_size)
        best_genome = np.argmax(self.generation_fitnesses)
        _, self.last_gens_best_genome = genomes[best_genome]
        print(f'Generation {self.cur_gen}: Avg fitness: {avg_gen_fit} Avg size: {avg_gen_size}')
        print(f'Generation {self.cur_gen}: Best genome fitness: {self.generation_fitnesses[best_genome]} Best genome\'s size: {self.generation_size[best_genome]}')
        self.generation_fitnesses = []
        self.generation_size = []
        self.cur_gen += 1



def run(n_generations, cfg_name="neat.cfg"):
    batch_size = 32

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), cfg_name)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    eval_genomes = GenomeEvaluator(batch_size)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("neat.log", eval_genomes)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)

    avg_fitness = eval_genomes.avg_fitness
    avg_size = eval_genomes.avg_size
    best_genome = eval_genomes.last_gens_best_genome

    save_name_suffix = cfg_name.split('.')[0]
    print('avg_fitness', avg_fitness)
    print('avg_size', avg_size)
    plt.plot(avg_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.savefig(f'average_fitnesses_rnn_{save_name_suffix}.png')

    plt.clf()

    plt.plot(avg_size)
    plt.xlabel('Generation')
    plt.ylabel('Average Number of Params')
    plt.savefig(f'average_params_rnn_{save_name_suffix}.png')

    plt.clf()

    # Now get the test accuracy on the best genome in this population
    phenotype = RecurrentNet.create(best_genome, config, batch_size)
    dataset = eval_genomes.train_dataset
    dataset.test()
    dataset.truncate(eval_genomes.max_words)
    dataset.use_subset(dataset.max_len())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=eval_genomes.batch_size, shuffle=True, num_workers=eval_genomes.num_workers, collate_fn=eval_genomes.collater.custom_collate_fn)

    correct = 0
    for audio_features, input_lengths, labels in dataloader:
        audio_features = audio_features.to('cpu', dtype=torch.float)
        activation_labels = torch.tensor(labels['act']).to('cpu')
        
        true_batch_size = audio_features.size(0)
        audio_features = audio_features.reshape(true_batch_size, -1) # squash dimensions together
        # We need to pad to maximum batch size because this neat implementation uses matrix multiplication
        # which should make the implementation run much faster, even without GPU acceleration, but requires
        # batch size to be consistent which it is not usually
        padding_needed = eval_genomes.batch_size - true_batch_size
        mask = eval_genomes.batch_size - padding_needed
        if padding_needed:
            padding_values = torch.zeros(padding_needed, audio_features.size(1))
            audio_features = torch.vstack([audio_features, padding_values])
        outputs = phenotype.activate(audio_features)
        outputs = outputs[:mask]
        outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        correct += (outputs == activation_labels).float().sum()

    accuracy = correct / len(dataset)

    print('Best genome in final generation test accuracy:', accuracy)


# Sometimes all the species die out (could possibly be fixed by the population size, but this code is quick so I'll also run an additional with 100 for this experiment)
# but still use a try catch 
# success = False
# while not success:
#     try:
#         print('Running with population size 50')
#         run(100)
#         success = True
#     except Exception as e:
#         print(e)
#         print('Complete extinction in population -- restarting experiment')
success = False
while not success:
    try:
        print('Running with population size 100')
        run(100, 'neat100.cfg')
        success = True
    except Exception as e:
        print(e)
        print('Complete extinction in population -- restarting experiment')

success = False
while not success:
    try:
        print('Running with population size 100')
        run(100, 'neat100_higher_thresh.cfg')
        success = True
    except Exception as e:
        print(e)
        print('Complete extinction in population -- restarting experiment')