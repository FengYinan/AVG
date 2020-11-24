from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import batch_and_drop_remainder
import numpy as np
import c3d_model
import i3d
import input_data
import tensorflow as tf
import os
import pandas as pd
import random
import input_frame
from l2_attack import CarliniL2

def _variable_on_gpu(name, shape, initializer):
  #with tf.device('/gpu:%d' % gpu_id):
  with tf.device('/gpu:2'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

class c3d:
    def __init__(self, function):
        self.image_size = 224
        self.num_channels = c3d_model.CHANNELS
        self.num_labels = 101
        self.model = function

    def predict(self, video):
        rgb_fc_out = self.model(video, reuse=True)

        return rgb_fc_out

class FGSM(object) :
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

        #
        self.trainB_dataset = glob('./dataset/{}/*/*'.format(self.dataset_name))
        self.trainA_dataset = glob('./dataset/{}/*'.format('ucf101/BrushingTeeth'))# + glob('./dataset/{}/*'.format('ucf101/ApplyLipstick'))
        #self.trainA_dataset = glob('./dataset/{}/*'.format('ucf101/Bowling'))
        #self.trainB_dataset = glob('./dataset/{}/*'.format(self.dataset_name + '/trainB'))
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
    # Classifier
    ##################################################################################

    def classify(self, x, reuse=False, scop="RGB"):
        with tf.variable_scope(scop, reuse=reuse) as var_scope:
            rgb_model = i3d.InceptionI3d(
                400, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                x, is_training=False, dropout_keep_prob=1)
            rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)
            rgb_fc_out = tf.layers.dense(
                rgb_logits_dropout, self.num_lable, use_bias=True)

            return rgb_fc_out

    def calc_gradients(self,images_placeholder,taret,norm_score):

        one_hot_grad = tf.one_hot(taret, self.num_lable)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=norm_score, labels=one_hot_grad)




        var_grad = tf.gradients(loss, images_placeholder)


        gradient_record = tf.sign(var_grad)

        gradient_records = tf.squeeze(gradient_record)

        return gradient_records
    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        self.tlab = tf.placeholder(tf.int32, shape=(self.batch_size), name='Generator_target_labele')
        self.olab = tf.placeholder(tf.int32, shape=(self.batch_size), name='Generator_original_labele')

        """ Test """
        self.test_video = tf.placeholder(tf.float32, [self.batch_size, self.num_clips, self.img_size, self.img_size, self.img_ch], name='test_video')

        self.logit_original = self.classify(self.test_video)
        self.lab_original = tf.argmax(self.logit_original, axis=1)


        self.test_fake = tf.placeholder(tf.float32, [self.batch_size, self.num_clips, self.img_size, self.img_size, self.img_ch], name='test_fake')

        self.test_fake_noise = self.test_fake - self.test_video

        self.test_logit = self.classify(self.test_fake, reuse=True)

        tf.global_variables_initializer().run()

        variable_map = {}
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == 'RGB':
                variable_map[variable.name.replace(':0', '')] = variable
        self.classify_saver = tf.train.Saver(variable_map)
        self.classify_saver.restore(self.sess, self.classify_model_name)

        model = c3d(self.classify)

        self.attackl = CarliniL2(self.sess, model, None, None, batch_size=self.batch_size,
                            num_clips=16, max_iterations=2000, confidence=0)
        self.sess.graph.finalize()




    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type)



    def test(self,exe, num):

        test_files = glob('./dataset/{}/*'.format(self.dataset_name))#testA
        #test_files = glob('F:/AVG/6TB/UCF101/*')  # testA
        #self.trainA_dataset = glob('./dataset/{}/*'.format('ucf101/small'))
        test_files.sort()

        #classify_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_name')
        #self.classify_saver = tf.train.Saver(classify_var)


        self.tlab_t = np.random.randint(1, 2, size=(1,), dtype=np.int32) * np.ones((self.batch_size), dtype=np.int32)

        fool_number = np.zeros(shape=(102))
        attack_number = np.zeros(shape=(102))
        counter = np.zeros(shape=(102))
        ml2 = np.zeros(shape=(102), dtype=np.float32)




        for i  in range(num,num+101) : # A -> B
            test_list = glob('{}/*'.format(test_files[i]))
            next_start_pos = 0
            print('Processing video class: ' + test_files[i])


            self.tlab_o = i * np.ones((self.batch_size), dtype=np.int32)

            all_steps = 1#int((len(test_list) - 1) / self.batch_size + 1)

            for step in range(all_steps):
                sample_image, next_start_pos, read_dirnames, valid_len, _, _ = \
                        input_data.read_clip_and_label(test_list, self.batch_size, num_frames_per_clip=self.num_clips, start_pos=next_start_pos, crop_size=self.img_size)


                adv = self.attackl.attack(sample_image, self.tlab_t, self.tlab_o)

                logit, noise = self.sess.run([self.test_logit, self.test_fake_noise],
                                             feed_dict={self.test_video: sample_image, self.tlab: self.tlab_t,
                                                        self.olab: self.tlab_o, self.test_fake: adv})

                out = np.argmax(logit,axis=1)
                print("lable %s" % out)

                fool = np.sum( (out != self.tlab_o)[:valid_len] )
                attack = np.sum( (out == self.tlab_t)[:valid_len] )
                l2 = np.sqrt( np.sum( np.square( noise[:valid_len] ) ) ) / (valid_len * self.num_clips * self.img_size * self.img_size * self.img_ch)
                fool_number[i] += fool
                fool_number[-1] += fool
                attack_number[i] += attack
                attack_number[-1] += attack
                ml2[i] = (ml2[i] * counter[i] + l2) / (counter[i] + valid_len)
                ml2[-1] = (ml2[-1] * counter[-1] + l2) / (counter[-1] + valid_len)
                counter[i] += valid_len
                counter[-1] += valid_len

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

        writer.save()


