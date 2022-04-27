from absl import app, flags, logging

from genetic_iemocap.training.codeepneat.iemocap_environment import IEMOCAPEnvironment

import tfne
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_integer('logging_level',
                     default=None, help='Integer parameter specifying the verbosity of the absl logging library')
flags.DEFINE_string('config_file',
                    default=None, help='String parameter specifying the file path to the configuration file used for '
                                       'the TFNE evolutionary process')
flags.DEFINE_string('backup_dir',
                    default=None, help='String parameter specifying the directory path to where the TFNE state backups '
                                       'should be saved to')
flags.DEFINE_integer('max_generations',
                     default=None, help='Integer parameter specifying the intended maximum number of generations the '
                                        'population should be evolved')
flags.DEFINE_float('max_fitness',
                   default=None, help='Float parameter specifying the fitness of the best genome at which point the '
                                      'evolutionary process should preemptively end')


def codeepneat_cifar10_example(_):
    """
    This Example evolves a CoDeepNEAT population on the CIFAR10 dataset for 50 generations. Subsequently the best
    genome is trained for a single epoch and its genotype and Tensorflow model are backed up.
    """
    # Set standard configuration specific to TFNE but not the neuroevolution process
    logging_level = logging.INFO
    config_file_path = './iemocap_config.cfg'
    backup_dir_path = './tfne_state_backups/'
    max_generations = 50
    max_fitness = None

    # Read in optionally supplied flags, changing the just set standard configuration
    if flags.FLAGS.logging_level is not None:
        logging_level = flags.FLAGS.logging_level
    if flags.FLAGS.config_file is not None:
        config_file_path = flags.FLAGS.config_file
    if flags.FLAGS.backup_dir is not None:
        backup_dir_path = flags.FLAGS.backup_dir
    if flags.FLAGS.max_generations is not None:
        max_generations = flags.FLAGS.max_generations
    if flags.FLAGS.max_fitness is not None:
        max_fitness = flags.FLAGS.max_fitness

    # Set logging, parse config
    logging.set_verbosity(logging_level)
    config = tfne.parse_configuration(config_file_path)

    # Initialize the environment and the specific NE algorithm
    environment = IEMOCAPEnvironment(weight_training=True, config=config, verbosity=logging_level)
    ne_algorithm = tfne.algorithms.CoDeepNEAT(config)

    # Initialize evolution engine and supply config as well as initialized NE algorithm and evaluation environment.
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=backup_dir_path,
                                  max_generations=max_generations,
                                  max_fitness=max_fitness)

    # Start training process, returning the best genome when training ends
    best_genome = engine.train()
    print("Best genome returned by evolution:\n")
    print(best_genome)

    # Increase epoch count in environment for a final training of the best genome. Train the genome and then replay it.
    print("Training best genome for 2 epochs...\n")
    environment.epochs = 2
    environment.eval_genome_fitness(best_genome)
    environment.replay_genome(best_genome)

    # Visualise the evolved genome
    best_genome.visualize(show=False, save_dir_path='./genome_images/')

    # Serialize and save genotype and Tensorflow model to demonstrate serialization
    best_genome.save_genotype(save_dir_path='./best_genome_genotype/')
    best_genome.save_model(file_path='./best_genome_model/')

    avg_fitness = ne_algorithm.avg_generation_fitness
    avg_size = ne_algorithm.avg_generation_size

    print('avg_fitness', avg_fitness)
    print('avg_size', avg_size)
    plt.plot(avg_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.savefig(f'average_fitnesses_codeepneat.png')

    plt.clf()

    plt.plot(avg_size)
    plt.xlabel('Generation')
    plt.ylabel('Average Number of Params')
    plt.savefig(f'average_params_codeepneat.png')

if __name__ == '__main__':
    app.run(codeepneat_cifar10_example)
