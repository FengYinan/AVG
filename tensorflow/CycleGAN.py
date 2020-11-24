from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import batch_and_drop_remainder
import numpy as np
import c3d_model
import input_data
import tensorflow as tf
import os
import pandas as pd
import random
import input_frame

def _variable_on_gpu(name, shape, initializer):
  #with tf.device('/gpu:%d' % gpu_id):
  with tf.device('/gpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

class CycleGAN(object) :
    def __init__(self, sess, args):
        self.model_name = 'CycleGAN'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag
        self.num_lable = args.num_lable
        self.train_list = args.train_list

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

        self.img_size = args.img_size
        self.num_clips = args.num_clips
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.ld = args.ld
        self.ch = args.ch
        
        self.classify_model_name = args.classify_model_name

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
        self.trainA_dataset = [('./dataset/{}/'.format(self.dataset_name) + i.strip('\n').split('.')[0]) for i in lines]
        #lable = [int(i.strip('\n').split()[1])-1 for i in lines]

        self.dataset_num = len(self.trainA_dataset)#max(, len(self.trainB_dataset))

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

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x, lable, reuse=False, scope="generator"):#
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self.ch
            embding = tf.get_variable(name='embding', initializer=np.random.randn(101, 5).astype(np.float32), trainable=True)
            c = tf.nn.embedding_lookup(embding, lable)
            c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, 1, c.shape[-1]]), tf.float32)
            c = tf.tile(c, [1, x.shape[1], x.shape[2], x.shape[3], 1])
            x = tf.concat([x, c], axis=-1)

            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            x = lrelu(x)

            # Down-Sampling
            for i in range(2) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, scope='conv_'+str(i+1))
                x = instance_norm(x, scope='down_ins_'+str(i+1))
                x = lrelu(x)

                channel = channel * 2


            for i in range(2) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, scope='conv_'+str(i+2+1))
                x = instance_norm(x, scope='down_ins_'+str(i+2+1))
                x = lrelu(x)

                channel = channel * 2

            # Bottle-neck
            for i in range(self.n_res) :
                x = resblock(x, channel, scope='resblock_'+str(i))

            #change video to image
            x = x[:, 0, :]

            # Up-Sampling
            for i in range(4):
                x = tf.image.resize_images(x, [x.shape[1] * 4, x.shape[2] * 4], method=1)
                x = conv2(x, channel // 2, kernel=3, stride=2, pad=1, scope='deconv' + str(i + 1))
                x = instance_norm(x, scope='up_ins_' + str(i + 1))
                x = lrelu(x)

                channel = channel // 2

            x = conv2(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')  #
            x = tanh(x)


            weight = tf.get_variable(name="weight", dtype=tf.float32, initializer=10., trainable=True)

            x = tf.reshape(x, shape=(self.batch_size, 1, x.shape[1], x.shape[2], x.shape[3])) * weight


            return x, c

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x, c, reuse=False, scope="discriminator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = tf.concat([x, c], axis=-1)

            x = conv(x, channel, kernel=3, stride=2, pad=1, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, scope='conv_'+str(i))
                x = instance_norm(x, scope='ins_'+str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = conv(x, channel*2, kernel=3, stride=1, pad=1, scope='conv_'+str(self.n_dis))
            x = instance_norm(x, scope='ins_'+str(self.n_dis))
            x = lrelu(x, 0.2)

            #x = conv(x, 1, kernel=4, stride=1, pad=1, scope='conv_' + str(self.n_dis+1))
            #x = instance_norm(x, scope='ins_' + str(self.n_dis+1))
            #x = lrelu(x, 0.2)


            x = conv(x, channels=1, kernel=3, stride=1, pad=1, scope='D_logit')
            #x = tf.transpose(x, perm=[0, 1, 4, 2, 3])
            #x = tf.reshape(x, [self.batch_size, x.get_shape().as_list()[-1] ** 2])
            #tf.contrib.layers.fully_connected(dense1, 1, activation_fn=tf.nn.softmax, scope="D_logit")

            return x


    def discriminator_black(self, x, reuse=False, scope="black"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=3, stride=1, pad=1, scope='conv_0')
            x = lrelu(x, 0.2)

            x=tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='pool_0')

            channel = channel * 2

            x = conv(x, channel, kernel=3, stride=1, pad=1, scope='conv_1')
            x = lrelu(x, 0.2)

            x=tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool_1')

            for i in range(1, self.n_dis) :
                x = conv(x, channel*2, kernel=3, stride=1, pad=1, scope='conv_'+str(1+i) + '_a')
                x = lrelu(x, 0.2)

                #x = conv(x, channel * 2, kernel=3, stride=1, pad=1, scope='conv_'+str(1+i) + '_b')
                #x = lrelu(x, 0.2)

                x=tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool_2')

                channel = channel * 2

            x = conv(x, channel, kernel=3, stride=1, pad=1, scope='conv_'+ str(self.n_dis + 1))
            x = lrelu(x, 0.2)

            #x = conv(x, channel, kernel=3, stride=1, pad=1, scope='conv_'+ str(self.n_dis + 2))
            #x = lrelu(x, 0.2)

            x=tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool_3')

            x = tf.transpose(x, perm=[0, 1, 4, 2, 3])
            x = tf.reshape(x, [8, 8192])
            x = tf.layers.dense(x, 4096, name='fc1')
            x = lrelu(x, 0.2)

            x = tf.layers.dense(x, 4096, name='fc2')
            x = lrelu(x, 0.2)

            x = tf.layers.dense(x, 101, name='out')

            return  x


    ##################################################################################
    # Classifier
    ##################################################################################

    def classify(self, x, reuse=False, scop="var_name"):
        with tf.variable_scope(scop, reuse=reuse) as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
                'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
                'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }

            logit = c3d_model.inference_c3d(x,
                                            0.6,
                                            self.batch_size,
                                            weights,
                                            biases)

            return logit

    ##################################################################################
    # Model
    ##################################################################################

    def gradient_panalty(self, real, fake, c, scope="discriminator"):
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1,1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake


        logit = self.discriminator(interpolated, c, reuse=True, scope=scope)


        GP = 0

        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

        # WGAN - LP
        if self.gan_type == 'wgan-lp' :
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP



    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.tlab = tf.placeholder(tf.int32, shape=(self.batch_size), name='Generator_target_labele')
        self.olab = tf.placeholder(tf.int32, shape=(self.batch_size), name='Generator_original_labele')

        """ Input Image"""
        self.domain_A = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_clips, self.img_size, self.img_size, self.img_ch))#trainA_iterator.get_next()
        #self.domain_B = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_clips, self.img_size, self.img_size, self.img_ch))

        self.logit_original = self.classify(self.domain_A)
        self.lab_original = tf.argmax(self.logit_original, axis=1)

        """ Define Encoder, Generator, Discriminator """
        x_fnoise, c = self.generator(self.domain_A, self.olab)
        x_fake = self.domain_A + tf.tile(x_fnoise, multiples=[1, 16, 1, 1, 1])
        x_fake = tf.minimum(tf.maximum(x_fake, 0.), 255.)

        self.cls_logit = self.classify(x_fake, reuse=True)

        #x_rnoise = self.generator(x_fake, self.olab, reuse=True)
        #x_recon = x_fake + tf.tile(x_rnoise, multiples=[1, 16, 1, 1, 1])
        #x_recon = tf.minimum(tf.maximum(x_recon, 0.), 255.)

        #self.recon_cls_logit = self.classify(x_recon, reuse=True)

        real_logit = self.discriminator(self.domain_A, c)
        fake_logit = self.discriminator(x_fake, c, reuse = True)

        """ Define Loss """
        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
            GP = self.gradient_panalty(real=self.domain_A, fake=x_fake, c=c)
        else :
            GP = 0

        self.G_ad_loss = generator_loss(self.gan_type, fake_logit)

        #self.recon_loss = adv_loss(self.recon_cls_logit, self.olab)#L1_loss(x_recon, self.domain_A)  # reconstruction

        self.l2_loss = L2_loss(x_fake, self.domain_A, self.l2_confidence)

        D_ad_loss = discriminator_loss(self.gan_type, real_logit, fake_logit) + GP

        if self.black_box:
            real_cls = self.discriminator_black(self.domain_A)
            fake_cls = self.discriminator_black(x_fake, reuse=True)

            self.cls_loss = adv_loss(fake_cls, self.tlab)

            hard_lable_real = tf.nn.softmax(self.logit_original)
            #soft_lable_real = tf.nn.softmax(self.logit_original/self.temperature)
            #lable_real = 0.1 * hard_lable_real + 0.9* soft_lable_real

            hard_lable_fake = tf.nn.softmax(self.cls_logit)
            #soft_lable_fake = tf.nn.softmax(self.cls_logit / self.temperature)
            #lable_fake = 0.1 * hard_lable_fake + 0.9 * soft_lable_fake

            #k = tf.constant([1 if k == 1 or k == 20 else 0 for k in range(101)], dtype=tf.float32)
            #k = tf.reshape(k,(1,101))
            #hard_lable_real = hard_lable_real*k
            #real_cls=real_cls*k
            #hard_lable_fake=hard_lable_fake*k
            #fake_cls=fake_cls*k

            self.B_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = hard_lable_real, logits= real_cls/self.temperature)) + \
                         tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = hard_lable_fake, logits= fake_cls/self.temperature))


        else:
            self.cls_loss = adv_loss(self.cls_logit, self.tlab)




        self.Generator_loss = self.gan_w * self.G_ad_loss + \
                           self.l2_w * self.l2_loss + \
                           self.identity_w * self.cls_loss #+ \
        #self.cycle_w * self.recon_loss


        self.Discriminator_loss = self.gan_w * D_ad_loss


        """ Training """
        t_vars = tf.trainable_variables()
        self.G_vars = [var for var in t_vars if 'generator' in var.name]
        self.D_vars = [var for var in t_vars if 'discriminator' in var.name]


        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=self.G_vars) #
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=self.D_vars) #


        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)


        self.g_adv_loss = tf.summary.scalar("g_adv_loss", self.G_ad_loss)
        self.g_cls_loss = tf.summary.scalar("cls_loss", self.cls_loss)
        #self.g_rec_loss = tf.summary.scalar("recon_loss", self.recon_loss)
        self.g_l2_loss = tf.summary.scalar("l2_loss", self.l2_loss)

        self.d_adv_loss = tf.summary.scalar("d_adv_loss", D_ad_loss)



        self.G_loss = tf.summary.merge([self.all_G_loss, self.g_adv_loss, self.g_cls_loss, self.g_l2_loss]) #self.g_rec_loss,
        self.D_loss = tf.summary.merge([self.all_D_loss, self.d_adv_loss])

        """" Black """
        if self.black_box:
            self.B_vars = [var for var in t_vars if 'black' in var.name]
            self.B_optim = tf.train.AdamOptimizer(self.lr/10, beta1=0.5, beta2=0.999).minimize(self.B_cls_loss, var_list=self.B_vars)
            self.all_B_loss = tf.summary.scalar("Black_loss", self.B_cls_loss)
            self.B_loss = tf.summary.merge([self.all_B_loss])


        """ Video """
        self.fake = x_fake

        self.real = self.domain_A

        #self.recon = x_recon


        """ Test """
        self.test_video = tf.placeholder(tf.float32, [self.batch_size, self.num_clips, self.img_size, self.img_size, self.img_ch], name='test_video')
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.num_clips, self.img_size, self.img_size, self.img_ch], name='mask')

        self.test_fake_noise, _ = self.generator(self.test_video, self.olab, reuse=True)
        self.test_fake = self.test_video + tf.tile(self.test_fake_noise, multiples=[1, 16, 1, 1, 1]) * self.mask
        self.test_fake = tf.minimum(tf.maximum(self.test_fake , 0.), 255.)

        self.test_logit = self.classify(self.test_fake, reuse=True)


    def train(self):
        classify_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_name')
        self.classify_saver = tf.train.Saver(classify_var)

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(self.G_vars + self.D_vars)

        if self.black_box:
            self.saver_black = tf.train.Saver(self.B_vars)

        self.classify_saver.restore(self.sess, self.classify_model_name)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        if self.black_box:
            _, _ = self.load_black(self.checkpoint_dir)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_d_loss = -1.
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag :
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch) # linear decay

            next_start_pos_a = 0
            next_start_pos_b = 0

            for idx in range(start_batch_id, self.iteration):

                if idx*self.batch_size%len(self.trainA_dataset) == 0:
                    next_start_pos_a = 0
                    next_start_pos_b = 0

                if (self.black_box and epoch % 2 != 0) or not self.black_box:
                    video, next_start_pos_a, read_dirnames, _, _, _ = \
                        input_data.read_clip_and_label(self.trainA_dataset, self.batch_size,
                                                       num_frames_per_clip=self.num_clips, start_pos=next_start_pos_a)

                if self.black_box and epoch % 2 == 0:
                    video, next_start_pos_b, read_dirnames, _, _, _ = \
                        input_data.read_clip_and_label(self.trainB_dataset, self.batch_size,
                                                       num_frames_per_clip=self.num_clips,
                                                       start_pos=next_start_pos_b)




                if video.shape[1] > 16:
                    video = video[:,0:16,]

                """Target class"""
                self.tlab_o = self.sess.run(self.lab_original, feed_dict = {self.domain_A : video})
                self.tlab_t = np.random.randint(66,67,size=(1,),dtype=np.int32) * np.ones((self.batch_size), dtype=np.int32)

                train_feed_dict = {
                    self.lr : lr,
                    self.tlab : self.tlab_t,
                    self.olab : self.tlab_o,
                    self.domain_A : video,
                }

                if (self.black_box and epoch % 2 == 0):
                    _, b_loss, summary_str, g_loss, gad_loss, gcls_loss, gl2_loss, cls_logit = self.sess.run(
                        [self.B_optim, self.B_cls_loss, self.B_loss, self.Generator_loss, self.G_ad_loss, self.cls_loss,
                         self.l2_loss, self.cls_logit], feed_dict=train_feed_dict)

                    self.writer.add_summary(summary_str, counter)

                # Update D
                d_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_d_loss = d_loss

                # display training status
                counter += 1
                if d_loss == None:
                    d_loss = past_d_loss

                # Update G
                if (self.black_box and epoch % 2 != 0) or not self.black_box:
                    _, batch_images, fake, g_loss, summary_str, gad_loss, gcls_loss, gl2_loss, cls_logit = \
                        self.sess.run([self.G_optim, self.real, self.fake, self.Generator_loss, self.G_loss,
                                       self.G_ad_loss, self.cls_loss, self.l2_loss, self.cls_logit],
                                      feed_dict=train_feed_dict)  # self.recon_loss,#, greloss,

                    self.writer.add_summary(summary_str, counter)


                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f, lable: %s" \
                          % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss, np.argmax(cls_logit, axis=1) ) )
                print("g_adv_loss: %.8f, g_cls_loss: %.8f, g_l2_loss: %.8f" \
                          % (gad_loss, gcls_loss, gl2_loss))#, greloss#, g_recon_loss: %.8f
                if self.black_box and epoch % 2 == 0:
                    print("b_cls_loss: %.8f" %b_loss)


                if ((epoch+1)%5 == 0) and np.mod(idx+1, self.print_freq) == 0 :
                    real, fake = self.sess.run([self.real, self.fake],#, self.recon, recon
                                                             feed_dict=train_feed_dict)

                    for index in range(self.num_clips):
                        save_images(fake[0,index], [self.batch_size, 1],
                                   './{}/fake_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1, index))
                        #save_images(recon[0, index], [self.batch_size, 1],
                                    #'./{}/recon_l{}_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, self.tlab_t[0],
                        #epoch + 1, idx + 1, index))
                        save_images((fake - real)[0, index], [self.batch_size, 1],
                                    './{}/noise_l{}_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, self.tlab_t[0],
                                                                                 epoch + 1, idx + 1, index))

                if np.mod(idx+1, self.save_freq) == 0 :
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

        if self.black_box:
            checkpoint_dir_black = os.path.join(checkpoint_dir, self.model_dir) +'_black'

            if not os.path.exists(checkpoint_dir_black):
                os.makedirs(checkpoint_dir_black)

            self.saver_black.save(self.sess, os.path.join(checkpoint_dir_black, self.model_name + '.model'), global_step=step)

        print("Saved.")

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_black(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir) +'_black'

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_black.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test_small(self,exe):
        list_frameT_clipF =[]
        list_frameF_clipT =[]
        tf.global_variables_initializer().run()
        #test_files = glob('./dataset/{}/*'.format(self.dataset_name))#testA
        test_files = glob('./dataset/{}/*'.format('ucf101/Bowling'))
        test_files.sort()

        classify_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_name')
        self.classify_saver = tf.train.Saver(classify_var)
        self.classify_saver.restore(self.sess, self.classify_model_name)

        self.saver = tf.train.Saver(self.G_vars + self.D_vars)
        if self.black_box:
            self.saver_black = tf.train.Saver(self.B_vars)
            _, _ = self.load_black(self.checkpoint_dir)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        self.tlab_t = np.random.randint(1, 2, size=(1,), dtype=np.int32) * np.ones((self.batch_size), dtype=np.int32)

        fool_number = np.zeros(shape=(17))
        attack_number = np.zeros(shape=(17))
        counter = np.zeros(shape=(1))
        ml2 = np.zeros(shape=(17), dtype=np.float32)

        next_start_pos = 0
        for i  in range(len(test_files)) : # A -> B

            print('Processing video class: ' + test_files[i])

            self.tlab_o = int(i//10) * np.ones((self.batch_size), dtype=np.int32)

            for step in range(2):
                sample_image2, _, _, _, _, _ = \
                    input_data.read_clip_and_label(test_files, 1, num_frames_per_clip=self.num_clips,
                                                   start_pos=next_start_pos)
                sample_image, next_start_pos, read_dirnames, valid_len, _, _ = \
                        input_frame.read_clip_and_label(test_files, step, self.batch_size-1, num_frames_per_clip=self.num_clips, start_pos=next_start_pos)
                sample_image = np.concatenate((sample_image[:valid_len],sample_image2), axis=0)

                fake_img, logit, noise = self.sess.run([self.test_fake, self.test_logit, self.test_fake_noise], feed_dict={self.test_video: sample_image, self.lr : self.init_lr,
                        self.tlab : self.tlab_t,
                        self.olab : self.tlab_o,
                        self.domain_A : sample_image})

                out = np.argmax(logit,axis=1)
                print("lable %s" % out)

                fool = (out != self.tlab_o)[:valid_len+1]
                attack = (out == self.tlab_t)[:valid_len+1]
                l2 = np.sqrt( np.sum( np.square( noise[:valid_len+1] ), axis=(1,2,3,4) ) ) / (valid_len * self.img_size * self.img_size * self.img_ch)
                fool_number[step * 8:(step+1)*8] += fool[:valid_len]
                fool_number[-1] += fool[-1]
                attack_number[step * 8:(step+1)*8] += attack[:valid_len]
                attack_number[-1] += attack[-1]
                ml2[step * 8:(step+1)*8] = (ml2[step * 8:(step+1)*8] * counter + l2[step * 8:(step+1)*8]) / (counter + valid_len)
                ml2[-1] = (ml2[-1] * counter + l2[-1]) / (counter + valid_len)
                counter += valid_len
                if out[-1] == self.tlab_o[0] and sum(out[:valid_len] == self.tlab_t[:valid_len]):
                    list_frameF_clipT.append(test_files[i])
                if out[-1] == self.tlab_t[0] and sum(out[:valid_len] == self.tlab_o[:valid_len]):
                    list_frameT_clipF.append(test_files[i])


            image_path = os.path.join(self.result_dir, '{0}'.format(os.path.basename(read_dirnames[0])))
            check_folder(image_path)
            for frame in range(self.num_clips):
                save_images(fake_img[0, frame], [self.batch_size, 1],
                            './{}/fake_{}_{:06d}.jpg'.format(image_path, i, frame))
                save_images(sample_image[0, frame], [self.batch_size, 1],
                            './{}/real_{}_{:06d}.jpg'.format(image_path, i, frame))
                save_images((fake_img - sample_image)[0, frame], [self.batch_size, 1],
                            './{}/noise_{}_{:06d}.jpg'.format(image_path, i, frame))

        writer = pd.ExcelWriter(exe)

        data_df = pd.DataFrame(fool_number.reshape(1, 17))
        # change the index and column name
        data_df.columns = [str(i + 1) if i != 101 else 'all' for i in range(17)]
        data_df.index = ['F']
        data_df.to_excel(writer, 'page_1', float_format='%.5f')

        data_df = pd.DataFrame(attack_number.reshape(1, 17))
        # change the index and column name
        data_df.index = ['A']
        data_df.to_excel(writer, 'page_1', float_format='%.5f', startrow=2, header=False)

        data_df = pd.DataFrame(ml2.reshape(1, 17))
        # change the index and column name
        data_df.index = ['L']
        data_df.to_excel(writer, 'page_1', float_format='%.5f', startrow=3, header=False)

        data_df = pd.DataFrame(counter.reshape(1, 1))
        # change the index and column name
        data_df.index = ['C']
        data_df.to_excel(writer, 'page_1', float_format='%.5f', startrow=4, header=False)

        writer.save()

        print(tuple(list_frameT_clipF))
        print(len(tuple(list_frameT_clipF)))
        print(tuple(list_frameF_clipT))
        print(len(tuple(list_frameF_clipT)))


    def test(self,exe,num):
        tf.global_variables_initializer().run()
        test_files = glob('./dataset/{}/*'.format(self.dataset_name))#testA
        #self.trainA_dataset = glob('./dataset/{}/*'.format('ucf101/small'))
        test_files.sort()

        classify_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_name')
        self.classify_saver = tf.train.Saver(classify_var)
        self.classify_saver.restore(self.sess, self.classify_model_name)

        self.saver = tf.train.Saver(self.G_vars + self.D_vars)
        if self.black_box:
            self.saver_black = tf.train.Saver(self.B_vars)
            _, _ = self.load_black(self.checkpoint_dir)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        self.tlab_t = np.random.randint(1, 2, size=(1,), dtype=np.int32) * np.ones((self.batch_size), dtype=np.int32)

        fool_number = np.zeros(shape=(102))
        attack_number = np.zeros(shape=(102))
        counter = np.zeros(shape=(102))
        ml2 = np.zeros(shape=(102), dtype=np.float32)
        li = np.zeros((1),dtype=np.float32)

        masknp = np.zeros([self.batch_size, self.num_clips, self.img_size, self.img_size, self.img_ch], np.float32)
        if num==0:
            for i in range(2):  # 2/4/8/2,3/16
                masknp[:, i] += 1
        elif num==1:
            for i in range(4):  # 2/4/8/2,3/16
                masknp[:, i] += 1
        elif num==2:
            for i in range(8):  # 2/4/8/2,3/16
                masknp[:, i] += 1
        elif num==3:
            for i in range(2):  # 2/4/8/2,3/16
                masknp[:, i] += 1
            for i in [-1,-2,-3]:  # 2/4/8/2,3/16
                masknp[:, i] += 1
        elif num==4:
            for i in range(12):  # 2/4/8/2,3/16
                masknp[:, i] += 1


        for i  in range(len(test_files)) : # A -> B
            test_list = glob('{}/*'.format(test_files[i]))
            next_start_pos = 0
            print('Processing video class: ' + test_files[i])

            self.tlab_o = i * np.ones((self.batch_size), dtype=np.int32)

            all_steps = int((len(test_list) - 1) / self.batch_size + 1)

            for step in range(all_steps):
                sample_image, next_start_pos, read_dirnames, valid_len, _, _ = \
                        input_data.read_clip_and_label(test_list, self.batch_size, num_frames_per_clip=self.num_clips, start_pos=next_start_pos)

                fake_img, logit, noise = self.sess.run([self.test_fake, self.test_logit, self.test_fake_noise], feed_dict={self.test_video: sample_image, self.lr : self.init_lr,
                        self.tlab : self.tlab_t,
                        self.olab : self.tlab_o,
                        self.domain_A : sample_image,
                        self.mask : masknp})

                out = np.argmax(logit,axis=1)
                print("lable %s" % out)

                fool = np.sum( (out != self.tlab_o)[:valid_len] )
                attack = np.sum( (out == self.tlab_t)[:valid_len] )
                l2 = np.sqrt( np.sum( np.square( noise[:valid_len] ) ) ) / (valid_len * self.img_size * self.img_size * self.img_ch)
                fool_number[i] += fool
                fool_number[-1] += fool
                attack_number[i] += attack
                attack_number[-1] += attack
                ml2[i] = (ml2[i] * counter[i] + l2) / (counter[i] + valid_len)
                ml2[-1] = (ml2[-1] * counter[-1] + l2) / (counter[-1] + valid_len)
                counter[i] += valid_len
                counter[-1] += valid_len
                max = np.max(np.abs(noise))
                li = max if max > li else li

#                image_path = os.path.join(self.result_dir, '{0}'.format(os.path.basename(read_dirnames[0])))
 #               check_folder(image_path)
  #              for frame in range(self.num_clips):
   #                 save_images(fake_img[-3, frame], [self.batch_size, 1],
    #                            './{}/fake_{}_{}_{:06d}.jpg'.format(image_path, i, step, frame))
     #               save_images(sample_image[-3, frame], [self.batch_size, 1],
      #                          './{}/real_{}_{}_{:06d}.jpg'.format(image_path, i, step, frame))
       #             save_images((fake_img - sample_image)[-3, frame], [self.batch_size, 1],
        #                        './{}/noise_{}_{}_{:06d}.jpg'.format(image_path, i, step, frame))

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


