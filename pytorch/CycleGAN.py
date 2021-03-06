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
        self.trainA_dataset2 = glob('{}/*'.format(test_files[1])) #+ glob('{}/*'.format(test_files[1]))

        test_files = glob('/home/google/{}/*'.format(self.dataset_name))
        test_files.sort()
        self.trainA_dataset3 = glob('{}/*'.format(test_files[0]))

        self.trainA_dataset = self.trainA_dataset2
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

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(create_graph = True, retain_graph = True, outputs=y,inputs=x,grad_outputs=weight, only_inputs = True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        if self.resume_iters:
            checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
            print('Loading the trained models from step {}...'.format(resume_iters))
            G_path = os.path.join(checkpoint_dir, '{}-G.ckpt'.format(resume_iters))
            D_path = os.path.join(checkpoint_dir, '{}-D.ckpt'.format(resume_iters))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
            if self.black_box :#and not self.new_start:
                B_path = os.path.join(checkpoint_dir, '{}-B.ckpt'.format(resume_iters))
                self.B.load_state_dict(torch.load(B_path, map_location=lambda storage, loc: storage))
                #net.load_state_dict(torch.load('./c3d.pickle'))
                #checkpoint = torch.load('./resnet-50-kinetics-ucf101_split1.pth')
                #self.B.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})


        checkpoint = torch.load(self.classify_model_name)

        # resxtnet101
        self.C.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
        #self.C.load_state_dict(checkpoint['state_dict'])

        #weights = torch.load(self.classify_model_name)['state_dict']
        #self.C.load_state_dict(weights)

        del checkpoint

    def save(self, save_dir, counter):
        self.model_save_dir = os.path.join(save_dir, self.model_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(counter + 1))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(counter + 1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)

        if self.black_box:
            B_path = os.path.join(self.model_save_dir, '{}-B.ckpt'.format(counter + 1))
            torch.save(self.B.state_dict(), B_path)

        print('Saved model {} checkpoints into {}...'.format(counter+1, self.model_save_dir))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)

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

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr
        if self.black_box:
            for param_group in self.b_optimizer.param_groups:
                param_group['lr'] = lr

    def B_update_lr(self, lr):
        for param_group in self.b_optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        if self.black_box:
            self.b_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def build_model(self):
        self.G = Generator(self.ch, self.embeding, self.num_lable, self.n_res)
        self.D = Discriminator(self.ch, self.embeding)
        self.C = Classifier(num_classes=self.num_lable, shortcut_type='B', cardinality=32, sample_size=self.img_size, sample_duration=self.num_clips)
        #Blackbox(num_classes=self.num_lable, shortcut_type='B', sample_size=self.img_size,sample_duration=self.num_clips)
        #

        if self.print_net:
            self.print_network(self.G, 'G')
            self.print_network(self.D, 'D')
            #self.print_network(self.C, 'C')
            if self.black_box:
                self.print_network(self.B, 'B')

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.init_lr, [0.5, 0.999])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.init_lr, [0.5, 0.999])

        if self.black_box:
            self.B = Blackbox(num_classes=self.num_lable, shortcut_type='B', sample_size=self.img_size,
                              sample_duration=self.num_clips)
            train_params = [{'params': C3D_model.get_1x_lr_params(self.B), 'lr': self.init_lr/1000},
                            {'params': C3D_model.get_10x_lr_params(self.B), 'lr': self.init_lr/100}]
            self.b_optimizer = torch.optim.Adam(train_params, self.init_lr, [0.5, 0.999])#torch.optim.SGD(train_params, lr=self.init_lr, momentum=0.9, weight_decay=5e-4)
                #torch.optim.SGD(self.B.parameters(), lr=self.init_lr, momentum=0.9, dampening=0.9, weight_decay=1e-3, nesterov=False)
                #

        self.G.to(self.device)
        self.D.to(self.device)
        # self.G = nn.DataParallel(self.G)
        # self.D = nn.DataParallel(self.D)
        # self.C = nn.DataParallel(self.C)
        if self.black_box:
            self.B.to(self.device)
            self.C.to(torch.device('cuda:0'))
        else:
            self.C.to(self.device)
            # self.B = nn.DataParallel(self.B)

    def train(self):
        start_iters = self.resume_iters if not self.new_start else 0
        self.restore_model(self.resume_iters)

        start_epoch = (int)(start_iters / self.iteration)
        start_batch_id = start_iters - start_epoch * self.iteration

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr

        train_G = True
        train_B = False

        self.G.train()
        self.D.train()
        self.C.eval()
        if self.black_box:
            self.B.train()
            self.set_requires_grad([self.B], False)

        self.tlab_t = np.array(1, dtype=np.int32) * np.ones((self.batch_size), dtype=np.int32)
        self.tlab_t = torch.from_numpy(self.tlab_t).to(self.device).requires_grad_(False)

        self.set_requires_grad([self.C], False)
        self.set_requires_grad([self.D], False)

        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag and epoch > self.decay_epoch:
                lr = self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch) # linear decay
                self.update_lr(lr)
            if self.black_box and self.decay_flag and epoch % 20 == 0:
                self.B_update_lr(self.init_lr / pow(10., (epoch/20) ) )

            next_start_pos_a = -1
            next_start_pos_b = -1


            #if (self.black_box and epoch % 2 == 0) or not self.black_box:
                #self.iteration = math.ceil(len(self.trainA_dataset) / self.batch_size)
            #elif self.black_box and epoch % 2 != 0:
                #self.iteration = math.ceil(len(self.trainB_dataset) / self.batch_size)
            #self.iteration = int(self.iteration)


            for idx in range(start_batch_id, self.iteration):

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                if len(self.trainA_dataset) - idx*self.batch_size < 0:
                    next_start_pos_a = -1
                    next_start_pos_b = -1

                if True:#(self.black_box and epoch % 2 == 0) or not self.black_box:
                    original, next_start_pos_a, read_dirnames, _, _, _ = \
                        input_data.read_clip_and_label(self.trainA_dataset, self.batch_size,
                                                       num_frames_per_clip=self.num_clips, start_pos=next_start_pos_a)

                    video =  np.zeros((self.batch_size,self.num_clips,self.img_size,self.img_size,3),dtype=np.float32) + 122.5

                    video += np.random.randn(self.batch_size, self.num_clips, self.img_size, self.img_size, 3)  #


                    if self.black_box:
                        if epoch % 3 < 5:
                            train_G = True
                            train_B = False
                        else:
                            train_G = False
                            train_B = True

                #else:# self.black_box and epoch % 2 != 0:
                    #video, next_start_pos_b, read_dirnames, _, _, _ = \
                        #input_data.read_clip_and_label(self.trainB_dataset, self.batch_size,
                                                       #num_frames_per_clip=self.num_clips,
                                                       #start_pos=next_start_pos_b)
                    #train_G = False
                    #train_B = True


                if original.shape[1] > 16:
                    original = original[:,0:16,:]
                video = np.pad(video, ((0, 0), (3, 3), (3, 3), (3, 3), (0, 0)), 'reflect')

                original = np.pad(original, ((0, 0), (3, 3), (3, 3), (3, 3), (0, 0)), 'reflect')
                original = torch.from_numpy(original.transpose((0, 4, 1, 2, 3))).float().to(self.device).requires_grad_(
                    False)



                if not self.black_box:
                    video = torch.from_numpy(video.transpose((0, 4, 1, 2, 3))).float().to(self.device).requires_grad_(
                        False)
                    self.output_real = self.C(video[:, :, 3:-3, 3:-3, 3:-3])
                else:
                    video = torch.from_numpy(video.transpose((0, 4, 1, 2, 3))).float().to(torch.device('cuda:0')).requires_grad_(
                        False)
                    self.output_real = self.C(video[:, :, 3:-3, 3:-3, 3:-3])
                    video =  video.to(self.device)
                    self.output_real = self.output_real.to(self.device)

                #self.tlab_o = np.array(0, dtype=np.int64) * np.ones((self.batch_size), dtype=np.int64)#
                self.tlab_o = torch.argmax(self.output_real, dim=1)
                #self.tlab_o = torch.from_numpy(self.tlab_o).to(self.device).requires_grad_(False)#
                self.tlab_o = self.tlab_o.to(self.device).requires_grad_(False)

                loss = {}

                # =================================================================================== #
                #                             2. Train the Generator                              #
                # =================================================================================== #
                if idx % self.n_critic == 0:#(self.black_box and epoch % 2 != 0) or not self.black_box:
                    if train_G:
                        self.set_requires_grad([self.G], True)
                    # Original-to-target domain.
                    x_fnoise, c = self.G(original, self.tlab_o)
                    video = video[:,:,3:-3,3:-3,3:-3]
                    original = original[:, :, 3:-3, 3:-3, 3:-3]
                    c = c[:,:,3:-3,3:-3,3:-3]
                    x_fake = video + x_fnoise.repeat(1, 1, 16, 1, 1)
                    x_fake = torch.clamp(x_fake, min=0., max=255.)



                    fake_logit = self.D(x_fake, c)

                    self.G_ad_loss = generator_loss(self.gan_type, fake_logit)
                    self.l2_loss = L2_loss(x_fake, video, self.l2_confidence)

                    if self.black_box:
                        fake_cls = self.B(x_fake)
                        self.cls_loss = adv_loss(fake_cls, self.tlab_t, black = True)
                        lable = torch.argmax(fake_cls, dim=1).requires_grad_(False)

                        del fake_cls
                    else:
                        self.cls_logit = self.C(x_fake)
                        self.cls_loss = adv_loss(self.cls_logit, self.tlab_t, num_labels = self.num_lable)
                        lable = torch.argmax(self.cls_logit, dim=1).requires_grad_(False)

                    self.cycle_loss = L2_loss(x_fake, original, 400)

                    self.Generator_loss = (self.identity_w * self.cls_loss
                                           + self.gan_w * self.G_ad_loss
                                           + self.cycle_w * self.cycle_loss)
                                         # self.l2_w * self.l2_loss  +


                    # Logging.
                    loss['G/loss_adv'] = self.G_ad_loss.item()
                    loss['G/loss_l2'] = self.l2_loss.item()
                    #loss['G/loss_cycle'] = self.cycle_loss.item()
                    loss['G/loss_cls'] = self.cls_loss.item()
                    loss['O/lable'] = self.tlab_o.cpu().detach().numpy()
                    if not self.black_box:
                        loss['G/lable'] = lable.cpu().detach().numpy()

                    del x_fnoise
                    del fake_logit
                    del self.G_ad_loss
                    del self.l2_loss
                    del self.cls_loss
                    del lable
                    torch.cuda.empty_cache()

                    # Backward and optimize.
                    if train_G:
                        self.reset_grad()
                        self.Generator_loss.backward()
                        self.g_optimizer.step()


                    del self.Generator_loss

                    self.set_requires_grad([self.G], False)

                # =================================================================================== #
                #                             2. Train the Blackbox                              #
                # =================================================================================== #
                if self.black_box and train_B:
                    self.set_requires_grad([self.B], True)

                    fake_cpu = x_fake.to(torch.device('cuda:0'))
                    self.cls_logit = self.C(fake_cpu.detach())
                    self.cls_logit = self.cls_logit.to(self.device)
                    lable = torch.argmax(self.cls_logit, dim=1).requires_grad_(False)

                    video =video.requires_grad_(True)
                    fake_cls = self.B(x_fake.detach())
                    real_cls = self.B(video)


                    J = self.compute_jacobian(video, real_cls)
                    new_video1 = video + 0.1 * torch.sign(J)
                    new_video1 = new_video1.detach().requires_grad_(True)
                    new_cls1 = self.B(new_video1)

                    J = self.compute_jacobian(new_video1, new_cls1)
                    new_video = new_video1 + 0.1 * torch.sign(J)
                    #new_video = new_video.detach().requires_grad_(True)
                    new_cls = self.B(new_video)


                    new_cpu = new_video.to(torch.device('cuda:0'))
                    new_output = self.C(new_cpu.detach())
                    new_output = new_output.to(self.device)

                    real_logit = torch.sigmoid(real_cls / self.temperature)
                    fake_logit = torch.sigmoid(fake_cls / self.temperature)
                    new_logit = torch.sigmoid(new_cls / self.temperature)


                    lable_real = F.softmax(self.output_real, dim=1).requires_grad_(False)
                    lable_fake = F.softmax(self.cls_logit, dim=1).requires_grad_(False)
                    lable_new = F.softmax(new_output, dim=1).requires_grad_(False)

                    lable_real, lable_fake = target_lable_B(lable_real, lable_fake, self.tlab_t)
                    lable_new, _ = target_lable_B(lable_new, lable_new, self.tlab_t)

                    self.Blackbox_loss = torch.mean(cross_entropy(lable=lable_real.detach(), output=real_logit)) + \
                                      torch.mean(cross_entropy(lable=lable_fake.detach(), output=fake_logit)) + \
                                      torch.mean(cross_entropy(lable=lable_new.detach(), output=new_logit))

                    if train_B:
                        self.reset_grad()
                        self.Blackbox_loss.backward()
                        self.b_optimizer.step()

                    del fake_cpu
                    del real_cls
                    del fake_cls
                    del J
                    del new_video
                    del new_video1
                    del self.cls_logit

                    del new_cls
                    del new_cpu
                    del new_output

                    del new_logit

                    del lable_new
                    torch.cuda.empty_cache()


                    loss['G/lable'] = lable.cpu().detach().numpy()
                    loss['B/loss'] = self.Blackbox_loss.item()
                    loss['B/O/real/cls'] = torch.mean(real_logit).cpu().detach().numpy()
                    loss['B/G/fake/cls'] = torch.mean(fake_logit).cpu().detach().numpy()
                    loss['C/O/real/cls'] = torch.mean(lable_real).cpu().detach().numpy()
                    loss['C/G/fake/cls'] = torch.mean(lable_fake).cpu().detach().numpy()

                    del self.Blackbox_loss
                    del lable
                    del real_logit
                    del fake_logit
                    del lable_fake
                    del lable_real

                    self.set_requires_grad([self.B], False)

                # =================================================================================== #
                #                             3. Train the Discriminator                              #
                # =================================================================================== #
                if True:
                    self.set_requires_grad([self.D], True)


                    if idx % self.n_critic != 0:
                        x_fnoise, c = self.G(video, self.tlab_o)
                        video = video[:, :, 3:-3, 3:-3, 3:-3]
                        c = c[:, :, 3:-3, 3:-3, 3:-3]
                        x_fake = video + x_fnoise.repeat(1, 1, 16, 1, 1)
                        x_fake = torch.clamp(x_fake, min=0., max=255.)

                    real_logit = self.D(original, c.detach())
                    fake_logit = self.D(x_fake.detach(), c.detach())

                    if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
                        alpha = torch.rand(video.size(0), 1, 1, 1, 1).to(self.device)
                        x_hat = (alpha * video.detach() + (1 - alpha) * x_fake.detach()).requires_grad_(True)
                        out_src = self.D(x_hat, c.detach())
                        GP = self.gradient_penalty(out_src, x_hat)

                        del alpha
                        del out_src
                        del x_hat
                    else:
                        GP = 0

                    # Backward and optimize.
                    D_ad_loss = discriminator_loss(self.gan_type, real_logit, fake_logit) + GP
                    self.Discriminator_loss = self.gan_w * D_ad_loss
                    self.reset_grad()
                    self.Discriminator_loss.backward()
                    self.d_optimizer.step()


                    del real_logit
                    del fake_logit
                    del GP
                    del D_ad_loss



                    # Logging.
                    loss['D/loss'] = self.Discriminator_loss.item()

                    del self.Discriminator_loss

                    self.set_requires_grad([self.D], False)

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #



                start_iters += 1

                # Print out training information.
                if idx  % self.print_freq == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}]".format(et, epoch+1, self.epoch, idx + 1, self.iteration)
                    for tag, value in loss.items():
                        if 'loss' in tag:# != 'G/lable' and tag !='O/lable':
                            log += ", {}: {:.4f}".format(tag, value)
                            #self.logger.scalar_summary(tag, value, start_iters)
                        elif 'cls' in tag:
                            log += ", {}: {:.4f}".format(tag, value)
                        else:
                            log += ", {}: {}".format(tag, value)
                    print(log)

                # Translate fixed images for debugging.

                if (idx + 1) % self.img_freq == 0:
                    with torch.no_grad():
                        x_fake = x_fake.cpu().detach().numpy().transpose((0, 2, 3, 4, 1))
                        video = video.cpu().detach().numpy().transpose((0, 2, 3 ,4, 1))
                        for index in range(self.num_clips):
                            save_images(x_fake[0, index], [self.batch_size, 1],'./{}/fake_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx + 1, index))
                            # save_images(recon[0, index], [self.batch_size, 1],
                            # './{}/recon_l{}_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, self.tlab_t[0],
                            # epoch + 1, idx + 1, index))
                            save_images((x_fake - video)[0, index], [self.batch_size, 1],'./{}/noise_l{}_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, self.tlab_t[0],
                                       epoch + 1, idx + 1, index))
                        print('Saved real and fake images into {}...'.format(self.sample_dir))

                del x_fake
                del c
                del video
                torch.cuda.empty_cache()

                # Save model checkpoints.
                if (idx + 1) % self.save_freq == 0:
                    self.save(self.checkpoint_dir, start_iters)
                    torch.cuda.empty_cache()

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, start_iters)
            torch.cuda.empty_cache()

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type)

    def test(self, exe):
        test_files = glob('/home/google/{}/*'.format(self.dataset_name))  # testA
        test_files.sort()

        self.restore_model(self.resume_iters)

        self.C.eval()
        self.G.train()
        del self.D
        if self.black_box:
            del self.B
        torch.cuda.empty_cache()
        self.C = self.C.to(self.device)

        self.set_requires_grad([self.C], False)
        self.set_requires_grad([self.G], False)


        fool_number = np.zeros(shape=(102))
        attack_number = np.zeros(shape=(102))
        counter = np.zeros(shape=(102))
        ml2 = np.zeros(shape=(102), dtype=np.float32)
        li = np.zeros((1), dtype=np.float32)

        tlab = np.array(1, dtype=np.int32) * np.ones((self.batch_size), dtype=np.int32)
        self.tlab_t = torch.from_numpy(tlab).to(self.device).requires_grad_(False)


        for idx in range(1):#len(test_files)) : # A -> B
            test_list = glob('{}/*'.format(test_files[idx]))
            next_start_pos = 0
            print('Processing video class: ' + test_files[idx])

            all_steps = 10#int((len(test_list) - 1) / self.batch_size + 1)

            for step in range(all_steps):
                #video, next_start_pos, read_dirnames, valid_len, _, _ = \
                    #input_data.read_clip_and_label(test_list, self.batch_size,
                                                   #num_frames_per_clip=self.num_clips, start_pos=next_start_pos)
                video = np.zeros((self.batch_size, self.num_clips, self.img_size, self.img_size, 3), dtype=np.float32) + 122.5
                valid_len =self.batch_size

                if video.shape[1] > 16:
                    video = video[:,0:16,:]
                video = np.pad(video, ((0, 0), (3, 3), (3, 3), (3, 3), (0, 0)), 'reflect')
                video = torch.from_numpy(video.transpose((0, 4, 1, 2, 3))).float().to(self.device).requires_grad_(False)

                """Target class"""
                self.output_real = self.C(video[:, :, 3:-3, 3:-3, 3:-3])
                self.tlab_o = torch.argmax(self.output_real, dim=1)

                self.tlab_o = self.tlab_o.to(self.device).requires_grad_(False)

                del self.output_real

                x_fnoise, _ = self.G(video, self.tlab_o)
                video = video[:, :, 3:-3, 3:-3, 3:-3]
                x_fake = video + x_fnoise.repeat(1, 1, 16, 1, 1)
                x_fake = torch.clamp(x_fake, min=0., max=255.)

                self.cls_logit = self.C(x_fake)
                lable = torch.argmax(self.cls_logit, dim=1).requires_grad_(False)

                out = lable.cpu().detach().numpy()
                olab = idx * np.ones((self.batch_size), dtype=np.int32)
                noise = x_fnoise.cpu().detach().numpy()

                print("lable %s" % out)

                fool = np.sum((out != olab)[:valid_len])
                attack = np.sum((out == tlab)[:valid_len])
                l2 = np.sqrt(np.sum(np.square(noise[:valid_len]))) / (
                            valid_len * self.img_size * self.img_size * self.img_ch)
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

                del x_fnoise
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