import torch
import torch.nn as nn
from torch import autograd
import neat_pytorch_linear
from neat_pytorch_linear.phenotype.feed_forward import FeedForwardNet
from genetic_iemocap.training.utils import CustomCollation
from genetic_iemocap.data_providers.iemocap import IEMOCAPDataset
import torch.nn.functional as F
import logging
import numpy as np
logger = logging.getLogger(__name__)



class IEMOCAPConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 5*10 # this is the size of the squashed input
    NUM_OUTPUTS = 3 # we bin activation down to low/medium/high
    USE_BIAS = True

    ACTIVATION = 'tanh'
    SCALE_ACTIVATION = 1

    FITNESS_THRESHOLD = 0.75

    POPULATION_SIZE = 50
    NUMBER_OF_GENERATIONS = 100
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.1
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30

    # Define the dataset for fitness evaluation
    # CONSTANTS 
    bs = 32
    num_workers = 2
    max_words = 5
    # END CONSTANTS 

    dataset = IEMOCAPDataset('../../data_providers/labels_dim10.pk', avg_audio_features=True, dim_size=10)
    dataset.truncate(max_words) # truncate dataset to only include utterances of length <= 10 words
    dataset.train()
    subset_size = 100
    dataset.use_subset(subset_size)

    collater = CustomCollation(max_len=max_words)

    cur_gen = 1
    generation_fitnesses = []
    generation_num_params = []
    avg_fitness = []
    avg_generation_num_params = []
    subset_size = 50

    def fitness_fn(self, genome, gen_check=True):
        if gen_check and self.cur_gen < genome.current_generation:
            self.cur_gen = genome.current_generation
            # Generation complete
            cur_gen_avg_fit = np.mean(self.generation_fitnesses)
            self.avg_fitness.append(cur_gen_avg_fit)
            self.avg_generation_num_params.append(np.mean(self.generation_num_params))
            logger.info(f'Generation{self.cur_gen} avg fitness: {self.avg_fitness[-1]} avg size: {self.avg_generation_num_params[-1]}')
            self.generation_fitnesses = []
            self.generation_num_params = []
            # use max in below calculations to ensure that we don't decrease if we got worse 
            old_size = self.subset_size
            if old_size < self.dataset.max_len():
                if cur_gen_avg_fit > 0.5 and self.subset_size == 1000:
                    self.subset_size = self.dataset.max_len()
                if cur_gen_avg_fit > 0.4 and self.subset_size == 200:
                    self.subset_size = max(min(1000, self.dataset.max_len()), self.subset_size)
                if cur_gen_avg_fit > 0.35: 
                    # Random guessing will be 0.33, so if we have improved beyond this increase the subset size
                    self.subset_size = max(min(200, self.dataset.max_len()), self.subset_size)
                if self.subset_size != old_size:
                    # log that fitness changed
                    logger.info(f'generation {self.cur_gen} increasing subset to {self.subset_size}')

        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        self.dataset.train()
        self.dataset.use_subset(self.subset_size)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.bs, shuffle=True, num_workers=self.num_workers, collate_fn=self.collater.custom_collate_fn)

        correct = 0
        with torch.no_grad():
            for audio_features, input_lengths, labels in dataloader:
                audio_features = audio_features.to(self.DEVICE, dtype=torch.float)
                activation_labels = torch.tensor(labels['act']).to(self.DEVICE)
                
                batch_size = audio_features.size(0)
                audio_features = audio_features.reshape(batch_size, -1) # squash dimensions together
                outputs = phenotype(audio_features)
                outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                correct += (outputs == activation_labels).float().sum()

        accuracy = correct / len(self.dataset)
        accuracy = accuracy.cpu().data

        self.generation_fitnesses.append(accuracy)
        num_params = sum(p.numel() for p in phenotype.parameters() if p.requires_grad)
        self.generation_num_params.append(num_params)
        logger.info(f'evaluated individual f: {accuracy} size: {num_params}')
        return accuracy # Accuracy will be the fitness for the phenotype

    def get_test_accuracy(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        self.dataset.test()
        self.dataset.use_subset(self.dataset.max_len())
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.bs, shuffle=True, num_workers=self.num_workers, collate_fn=self.collater.custom_collate_fn)

        correct = 0
        for audio_features, input_lengths, labels in dataloader:
            audio_features = audio_features.to(self.DEVICE, dtype=torch.float)
            activation_labels = torch.tensor(labels['act']).to(self.DEVICE)
            
            batch_size = audio_features.size(0)
            audio_features = audio_features.reshape(batch_size, -1) # squash dimensions together
            outputs = phenotype(audio_features)
            outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct += (outputs == activation_labels).float().sum()

        accuracy = correct / len(self.dataset)
        num_params = sum(p.numel() for p in phenotype.parameters() if p.requires_grad)

        return accuracy, num_params # Accuracy will be the fitness for the phenotype
