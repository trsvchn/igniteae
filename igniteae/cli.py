#!/usr/bin/env python3

import sys
import argparse
import logging
import main


def args_parser() -> dict:
    parser = argparse.ArgumentParser(description='Autoencoder Training.')

    parser.add_argument('--data_dir', default='./data', type=str, metavar='PATH',
                        help='Path to data. Default: ./data')

    parser.add_argument('-s', '--seed', default=42, type=int, metavar='SEED',
                        help='Random seed. Default: 42')

    parser.add_argument('-j', '--nworkers', default=2, type=int, metavar='N',
                        help='Number of data loading workers. Default: 2')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='Number of epochs to run. Default: 100')
    parser.add_argument('-p', '--print-every', default=1, type=int, metavar='N',
                        help='Print frequency (in number of epoch). Default: 1')

    parser.add_argument('-b', '--tbs', default=64, type=int, metavar='N',
                        help='Training batch size. Default: 64')
    parser.add_argument('-vb', '--vbs', default=64, type=int, metavar='N',
                        help='Validation batch size. Default: 64')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='Initial learning rate. Default: 0.001')
    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float, metavar='WD',
                        help='Weight decay. Default: 1e-5')

    parser.add_argument('-g', '--gpu', action='store_true',
                        help='Use GPU. Default: False')

    parser.add_argument('--reshuffle_data', action='store_true',
                        help='Reshuffle data at every epoch. Default: False')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    logging.info(f'Current interpreter: {sys.executable}')
    args_ = args_parser()
    logging.info('INPUT PARAMS:\n' + '\n'.join([': '.join([k, str(v)]) for k, v in args_.items()]))
    main.main(**args_)
