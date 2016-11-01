import _init_paths

import caffe
import argparse
import sys
import os.path as osp

#from configuration.config import cfg
from datasetfactory.ILSVRC import ILSVRC_handler
from datasetfactory.imdb import IMDB
from solver import train_net


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='weights',
                        help='pretrained model',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to train',
                        default=None, type=str)
    parser.add_argument('--out', dest='out',
                        help='output directory',
                        default='out', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    dataset = osp.join('data', args.dataset)
    print 'Dataset path: {}'.format(dataset)

    imdb = IMDB()
    imdb.load_data(ILSVRC_handler, dataset=dataset)
    print 'Data has all be loaded'

    output_dir = args.out
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, imdb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
