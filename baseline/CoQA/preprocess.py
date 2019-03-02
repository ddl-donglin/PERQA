import argparse
import logging
import multiprocessing
import random

import numpy as np
import spacy
from baseline.general_utils import load_glove_vocab


def main():
    parser = argparse.ArgumentParser(
        description='Preprocessing train + dev files, about 15 minutes to run on Servers.'
    )
    parser.add_argument('--wv_file', default='../glove/glove.840B.300d.txt',
                        help='path to word vector file.')
    parser.add_argument('--wv_dim', type=int, default=300,
                        help='word vector dimension.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words.'
                             'Otherwise consider question words first.')
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                        help='number of threads for preprocessing.')
    parser.add_argument('--no_match', action='store_true',
                        help='do not extract the three exact matching features.')
    parser.add_argument('--seed', type=int, default=1023,
                        help='random seed for data shuffling, embedding init, etc.')

    args = parser.parse_args()
    # trn_file = 'CoQA_data/train.json'
    # dev_file = 'CoQA_data/dev.json'
    wv_file = args.wv_file
    wv_dim = args.wv_dim
    nlp = spacy.load('en', disable=['parser'])

    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)

    log.info('start data preparing... (using {} threads)'.format(args.threads))

    glove_vocab = load_glove_vocab(wv_file, wv_dim)  # return a "set" of vocabulary
    log.info('glove loaded.')


if __name__ == '__main__':
    main()
