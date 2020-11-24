from CycleGAN_google import CycleGAN
import argparse
from utils import *
import os
#from torch.backends import cudnn
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_CHECKPOINT_PATHS = {
    'i3d': './ucf101_rgb_0.946_model-44520',
    'c3d': './sports1m_finetuning_ucf101.model',
    'i3d-kin': './rgb_imagenet/model.ckpt',
    'resnext-101-ucf101' : './resnext-101-kinetics-ucf101_split1.pth',
    'resnet-50-ucf101' : './resnet-50-kinetics-ucf101_split1.pth',
}

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of CycleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='hmdb51', help='dataset_name')
    parser.add_argument('--train_list', type=str, default='/home/google/CycleGAN/ucfTrainTestlist/trainlist01.txt', help='train_list')
    parser.add_argument('--test_list', type=str, default='/home/nesa320/yinan_2181200183/CycleGAN/ucfTrainTestlist/testlist01.txt', help='test_list')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--num_lable', type=int, default=101, help='num_lable')

    parser.add_argument('--black_box', type=bool, default=True, help='black_box')
    parser.add_argument('--temperature', type=float, default=1., help='decay temperature')




    parser.add_argument('--new_start', type=bool, default=False, help='decay new_start')
    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
    parser.add_argument('--modle_iters', type=int, default=0, help='resume training from this step')

    parser.add_argument('--decay_flag', type=bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=6, help='decay epoch')

    parser.add_argument('--epoch', type=int, default=80, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=2100, help='The number of training iterations')

    parser.add_argument('--lr', type=float, default=0.002, help='The learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')

    parser.add_argument('--gan_w', type=float, default=1.0, help='weight of adversarial loss')
    parser.add_argument('--l2_w', type=float, default=1e-4, help='weight of l2 loss')
    parser.add_argument('--cycle_w', type=float, default=0.0, help='weight of cycle loss')
    parser.add_argument('--identity_w', type=float, default=1.0, help='weight of identity loss')
    parser.add_argument('--l2_confidence', type=float, default=36., help='l2_confidence')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--embeding', type=int, default=1, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=6, help='The number of residual blocks')

    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=5, help='The number of ckpt_save_freq')
    parser.add_argument('--img_freq', type=int, default=200, help='The number of img_save_freq')
    parser.add_argument('--print_net', type=bool, default=False, help='print_net')




    parser.add_argument('--gan_type', type=str, default='wgan-gp', help='GAN loss type [gan / lsgan/ wgan-gp / wgan-lp / dragan / hinge]')

    parser.add_argument('--lab_o', type=str, default='hair', help='lable original')
    parser.add_argument('--lab_t', type=str, default='gun', help='lable target')

    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')

    parser.add_argument('--n_critic', type=int, default=5, help='The number of critic')

    parser.add_argument('--img_size', type=int, default=112, help='The size of image')
    parser.add_argument('--num_clips', type=int, default=16, help='The size of clip')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument('--video_dir', type=str, default='video',
                        help='Directory name to save the video on training')

    parser.add_argument('--classify_model_name', type=str, default='./resnext-101-kinetics-ucf101_split1.pth',
                        help='classify_model_name')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    check_folder(args.video_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    #cudnn.benchmark = True

    gan = CycleGAN(args)

    # build graph
    print(" [*] Buliding model!")
    gan.build_model()
    print(" [*] Training built model!")

    # show network architecture
    # show_all_variables()

    if args.phase == 'train':
        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        for i in range(1):
            gan.test('Results' + str(i + 1) + '.xlsx')
        print(" [*] Test finished!")


if __name__ == '__main__':
    main()