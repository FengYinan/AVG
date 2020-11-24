from ops import *
from utils import *
from glob import glob
import time
import datetime
import sys
import gc
import math

import os
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torchvision.utils import save_image
import numpy as np
from attack_carlini_wagner_l2 import AttackCarliniWagnerL2

import input_data

class CycleGAN(object) :
    def __init__(self, args):
        self.model_name = 'CycleGAN'
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag
        self.num_lable = args.num_lable
        self.train_list = args.train_list
        self.test_list = args.test_list

        self.black_box =args.black_box
        self.temperature = args.temperature

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.gan_type = args.gan_type
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.img_freq = args.img_freq

        self.img_size = args.img_size
        self.num_clips = args.num_clips
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.ld = args.ld
        self.ch = args.ch
        self.embeding = args.embeding

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.classify_model_name = args.classify_model_name

        self.print_net = args.print_net
        self.new_start = args.new_start
        self.resume_iters = args.resume_iters

        """ Target Lable"""
        self.tlab_t = args.lab_t
        self.tlab_o = args.lab_o

        """ Weight """
        self.gan_w = args.gan_w
        self.l2_w = args.l2_w
        self.identity_w = args.identity_w
        self.l2_confidence = args.l2_confidence
        self.cycle_w = args.cycle_w

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        #self.trainB_dataset = glob('./dataset/{}/*/*'.format(self.dataset_name))
        #self.trainA_dataset = glob('./dataset/{}/*'.format('ucf101/BrushingTeeth'))# + glob('./dataset/{}/*'.format('ucf101/ApplyLipstick'))
        #self.trainA_dataset = glob('./dataset/{}/*/*'.format(self.dataset_name))
        #self.trainB_dataset = glob('./dataset/{}/*'.format(self.dataset_name + '/trainB'))

        lines = list(open(self.train_list, 'r', encoding='UTF-8'))
        self.trainA_dataset1 = [('/home/google/{}/'.format(self.dataset_name) + i.strip('\n').split('.')[0]) for i in lines]
        #lable = [int(i.strip('\n').split()[1])-1 for i in lines]

        test_files = glob('/home/google/{}/*'.format(self.dataset_name))
        test_files.sort()
        self.trainA_dataset2 = glob('{}/*'.format(test_files[0])) #+ glob('{}/*'.format(test_files[1]))

        test_files = glob('/home/google/{}/*'.format(self.dataset_name))
        test_files.sort()
        self.trainA_dataset3 = glob('{}/*'.format(test_files[0]))

        self.trainA_dataset = self.trainA_dataset1
        #self.trainB_dataset = self.trainA_dataset1

        self.dataset_num = len(self.trainA_dataset)  # max(, len(self.trainB_dataset))

        #self.build_tensorboard()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)
        print("# the number of critic : ", self.n_critic)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""


        checkpoint = torch.load(self.classify_model_name)

        # resxtnet101
        self.C.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
        #self.C.load_state_dict(checkpoint['state_dict'])

        #weights = torch.load(self.classify_model_name)['state_dict']
        #self.C.load_state_dict(weights)

        del checkpoint

    def compute_jacobian(self, inputs, output):
        """
        :param inputs: Batch X Size (e.g. Depth X Width X Height)
        :param output: Batch X Classes
        :return: jacobian: Batch X Classes X Size
        """
        assert inputs.requires_grad

        num_classes = output.size()[1]

        jacobian = torch.zeros(num_classes, *inputs.size())
        grad_output = torch.zeros(*output.size())
        if inputs.is_cuda:
            grad_output = grad_output.cuda()
            jacobian = jacobian.cuda()

        for i in range(num_classes):
            zero_gradients(inputs)
            grad_output.zero_()
            grad_output[:, i] = 1
            output.backward(grad_output, retain_graph=True)
            jacobian[i] = inputs.grad.data

        return jacobian[0]

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def build_model(self):
        self.C = Classifier(num_classes=self.num_lable, shortcut_type='B', cardinality=32, sample_size=self.img_size, sample_duration=self.num_clips)
        #Blackbox(num_classes=self.num_lable, shortcut_type='B', sample_size=self.img_size,sample_duration=self.num_clips)

        self.C.to(self.device)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type)

    def test(self, exe):
        test_files = glob('/home/google/{}/*'.format(self.dataset_name))  # testA
        test_files.sort()

        self.restore_model(self.resume_iters)

        self.C.eval()

        self.C = self.C.to(self.device)

        self.set_requires_grad([self.C], False)


        fool_number = np.zeros(shape=(102))
        attack_number = np.zeros(shape=(102))
        counter = np.zeros(shape=(102))
        ml2 = np.zeros(shape=(102), dtype=np.float32)
        li = np.zeros((1), dtype=np.float32)

        tlab = np.array(1, dtype=np.int64) * np.ones((self.batch_size), dtype=np.int64)
        self.tlab_t = torch.from_numpy(tlab).to(self.device).requires_grad_(False)

        cwattack = AttackCarliniWagnerL2(
            targeted=True,
            max_steps=2000,
            search_steps=1,
            cuda=True,
            debug=False)


        for idx in range(92,93):#len(test_files)) : # A -> B
            test_list = glob('{}/*'.format(test_files[idx]))
            next_start_pos = 0
            print('Processing video class: ' + test_files[idx])

            all_steps = 1#int((len(test_list) - 1) / self.batch_size + 1)

            for step in range(all_steps):
                video, next_start_pos, read_dirnames, valid_len, _, _ = \
                    input_data.read_clip_and_label(test_list, self.batch_size,
                                                num_frames_per_clip=self.num_clips, start_pos=next_start_pos)
                #video = np.zeros((self.batch_size, self.num_clips, self.img_size, self.img_size, 3), dtype=np.float32) + 122.5
                #valid_len =self.batch_size

                if video.shape[1] > 16:
                    video = video[:,0:16,:]
                video = torch.from_numpy(video.transpose((0, 4, 1, 2, 3))).float().to(self.device).requires_grad_(True)

                """Target class"""
                self.output_real = self.C(video)
                self.tlab_o = np.array(idx, dtype=np.int64) * np.ones((self.batch_size), dtype=np.int64)

                self.tlab_o = torch.from_numpy(self.tlab_o).to(self.device).requires_grad_(False)

                original = video.cpu().detach().numpy()

                x_fake = cwattack.run(self.C, video, self.tlab_t, idx+1)
                x_fake = torch.clamp(x_fake, min=0., max=255.)

                self.cls_logit = self.C(x_fake)
                lable = torch.argmax(self.cls_logit, dim=1).requires_grad_(False)

                out = lable.cpu().detach().numpy()
                olab = idx * np.ones((self.batch_size), dtype=np.int32)
                noise = (x_fake-video).cpu().detach().numpy()

                print(np.sqrt(np.sum(np.square(noise-original))) / (
                            valid_len * self.img_size * self.img_size * self.img_ch*self.num_clips))


                print("lable %s" % out)

                fool = np.sum((out != olab)[:valid_len])
                attack = np.sum((out == tlab)[:valid_len])
                l2 = np.sqrt(np.sum(np.square(noise[:valid_len]))) / (
                            valid_len * self.img_size * self.img_size * self.img_ch*self.num_clips)
                fool_number[idx] += fool
                fool_number[-1] += fool
                attack_number[idx] += attack
                attack_number[-1] += attack
                ml2[idx] = (ml2[idx] * counter[idx] + l2) / (counter[idx] + valid_len)
                ml2[-1] = (ml2[-1] * counter[-1] + l2) / (counter[-1] + valid_len)
                counter[idx] += valid_len
                counter[-1] += valid_len
                max = np.max(np.abs(noise))
                li = max if max > li else li

                with torch.no_grad():
                    x_fake = x_fake.cpu().detach().numpy().transpose((0, 2, 3, 4, 1))
                    video = video.cpu().detach().numpy().transpose((0, 2, 3, 4, 1))
                    for index in range(self.num_clips):
                        save_images(x_fake[0, index], [self.batch_size, 1],
                                    './{}/fake_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, idx, step + 1, index))
                        # save_images(recon[0, index], [self.batch_size, 1],
                        # './{}/recon_l{}_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, self.tlab_t[0],
                        # epoch + 1, idx + 1, index))
                    save_images((x_fake - video)[0, index], [self.batch_size, 1],
                                './{}/noise_l{}_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, self.tlab_t[0],
                                                                                idx + 1, step + 1, index))
                    print('Saved real and fake images into {}...'.format(self.sample_dir))


                del x_fake
                del lable
                del video

                torch.cuda.empty_cache()

        writer = pd.ExcelWriter(exe)

        data_df = pd.DataFrame(fool_number.reshape(1, 102))
        # change the index and column name
        data_df.columns = [str(i + 1) if i != 101 else 'all' for i in range(102)]
        data_df.index = ['F']
        data_df.to_excel(writer, 'page_1', float_format='%.5f')

        data_df = pd.DataFrame(attack_number.reshape(1, 102))
        # change the index and column name
        data_df.index = ['A']
        data_df.to_excel(writer, 'page_1', float_format='%.5f', startrow=2, header=False)

        data_df = pd.DataFrame(ml2.reshape(1, 102))
        # change the index and column name
        data_df.index = ['L']
        data_df.to_excel(writer, 'page_1', float_format='%.5f', startrow=3, header=False)

        data_df = pd.DataFrame(counter.reshape(1, 102))
        # change the index and column name
        data_df.index = ['C']
        data_df.to_excel(writer, 'page_1', float_format='%.5f', startrow=4, header=False)

        data_df = pd.DataFrame(li.reshape(1, 1))
        # change the index and column name
        data_df.index = ['L']
        data_df.to_excel(writer, 'page_1', float_format='%.5f', startrow=5, header=False)

        writer.save()