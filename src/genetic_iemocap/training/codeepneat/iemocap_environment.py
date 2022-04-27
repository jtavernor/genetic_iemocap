from __future__ import annotations

import numpy as np
import tensorflow as tf

from tfne.environments.base_environment import BaseEnvironment
from tfne.helper_functions import read_option_from_config

from genetic_iemocap.training.codeepneat.tf_iemocap_dataset import convert_iemocap_to_tensorflow
from genetic_iemocap.data_providers.iemocap import IEMOCAPDataset

# This is a modified version of the cifar10 environment in the tfne code -- as can be seen from the comments
class IEMOCAPEnvironment(BaseEnvironment):
    """
    TFNE compatible environment for the CIFAR10 dataset
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self, weight_training, config=None, verbosity=0, **kwargs):
        """
        Initializes CIFAR10 environment by setting up the dataset and processing the supplied config or supplied config
        parameters. The configuration of the environment can either be supplied via a config file or via seperate config
        parameters in the initialization.
        @param weight_training: bool flag, indicating wether evaluation should be weight training or not
        @param config: ConfigParser instance holding an 'Environment' section specifying the required environment
                       parameters for the chosen evaluation method.
        @param verbosity: integer specifying the verbosity of the evaluation
        @param kwargs: Optionally supplied dict of each configuration parameter seperately in order to allow the
                       creation of the evaluation environment without the requirement of a config file.
        """
        # Load test data, unpack it and normalize the pixel values
        print("Setting up IEMOCAP environment...")
        dataset = IEMOCAPDataset('../../data_providers/labels_dim10.pk', avg_audio_features=True, dim_size=10)
        dataset.train()
        # No way to do a data subset for this one because I don't know enough about tensorflow
        max_words = 5
        dataset.truncate(max_words)
        (self.train_images, self.train_labels), (self.test_images, test_labels) = convert_iemocap_to_tensorflow(dataset, max_words, dim_size=10)
        # cifar10_dataset = tf.keras.datasets.cifar10.load_data()
        # (self.train_images, self.train_labels), (self.test_images, test_labels) = cifar10_dataset
        # self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
        self.squeezed_train_labels = np.squeeze(self.train_labels)
        self.squeezed_test_labels = np.squeeze(test_labels)

        # Initialize the accuracy metric, responsible for fitness determination and safe the verbosity parameter
        self.accuracy_metric = tf.keras.metrics.Accuracy()
        self.verbosity = verbosity

        self.avg_accuracy_sizes = []
        self.avg_model_sizes = []

        # Determine and setup explicit evaluation method in accordance to supplied parameters
        if not weight_training:
            raise NotImplementedError("CIFAR10 environment is being set up as non-weight training, though non-weight "
                                      "training evaluation not yet implemented for CIFAR10 environment")

        elif config is None and len(kwargs) == 0:
            raise RuntimeError("CIFAR10 environment is being set up as weight training, though neither config file nor "
                               "explicit config parameters for the weight training were supplied")

        elif len(kwargs) == 0:
            # Set up CIFAR10 environment as weight training and with a supplied config file
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training
            self.epochs = read_option_from_config(config, 'EVALUATION', 'epochs')
            self.batch_size = read_option_from_config(config, 'EVALUATION', 'batch_size')

        elif config is None:
            # Set up CIFAR10 environment as weight training and explicitely supplied parameters
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training
            self.epochs = kwargs['epochs']
            self.batch_size = kwargs['batch_size']

    def eval_genome_fitness(self, genome) -> float:
        # TO BE OVERRIDEN
        raise RuntimeError()

    def _eval_genome_fitness_weight_training(self, genome) -> float:
        """
        Evaluates the genome's fitness by obtaining the associated Tensorflow model and optimizer, compiling them and
        then training them for the config specified duration. The genomes fitness is then calculated and returned as
        the percentage of test images classified correctly.
        @param genome: TFNE compatible genome that is to be evaluated
        @return: genome calculated fitness that is the percentage of test images classified correctly
        """
        # Get model and optimizer required for compilation
        model = genome.get_model()
        optimizer = genome.get_optimizer()

        # Compile and train model
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
        model.fit(x=self.train_images,
                  y=self.train_labels,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  verbose=self.verbosity)

        # Determine fitness by creating model predictions with test images and then judging the fitness based on the
        # achieved model accuracy. Return this fitness
        self.accuracy_metric.reset_states()
        self.accuracy_metric.update_state(self.squeezed_train_labels, np.argmax(model(self.train_images), axis=-1))
        return round(self.accuracy_metric.result().numpy() * 100, 4)

    def _eval_genome_fitness_non_weight_training(self, genome) -> float:
        """"""
        raise NotImplementedError("Non-Weight training evaluation not yet implemented for CIFAR10 Environment")

    def replay_genome(self, genome):
        """
        Replay genome on environment by calculating its fitness and printing it. The fitness is the percentage of test
        images classified correctly.
        @param genome: TFNE compatible genome that is to be evaluated
        """
        print("Replaying Genome #{}:".format(genome.get_id()))

        # Determine fitness by creating model predictions with test images and then judging the fitness based on the
        # achieved model accuracy.
        model = genome.get_model()
        self.accuracy_metric.reset_states()
        self.accuracy_metric.update_state(self.squeezed_test_labels, np.argmax(model(self.test_images), axis=-1))
        evaluated_fitness = round(self.accuracy_metric.result().numpy() * 100, 4)
        print("Achieved Test Accuracy:\t{}\n".format(evaluated_fitness))
        model_size = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        print(f'Number of trainable parameters: {model_size}')

    def duplicate(self) -> IEMOCAPEnvironment:
        """
        @return: New instance of the XOR environment with identical parameters
        """
        if hasattr(self, 'epochs'):
            return IEMOCAPEnvironment(True, verbosity=self.verbosity, epochs=self.epochs, batch_size=self.batch_size)
        else:
            return IEMOCAPEnvironment(False, verbosity=self.verbosity)

    def get_input_shape(self) -> (int, int):
        """"""
        return 5, 10

    def get_output_shape(self) -> (int,):
        """"""
        return (3,)
