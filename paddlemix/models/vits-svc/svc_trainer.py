import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import paddle
# import torch.multiprocessing as mp
from omegaconf import OmegaConf

from vits_extend.train import train

# torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    assert hp.data.hop_length == 320, \
        'hp.data.hop_length must be equal to 320, got %d' % hp.data.hop_length

    args.num_gpus = 0
    paddle.seed(hp.train.seed)
    args.num_gpus = paddle.device.cuda.device_count()
    print('Batch size per GPU :', hp.train.batch_size)

    if args.num_gpus > 1:
        # mp.spawn(train, nprocs=args.num_gpus,
        #          args=(args, args.checkpoint_path, hp, hp_str,))
        pass
    else:
        train(0, args, args.checkpoint_path, hp, hp_str)

