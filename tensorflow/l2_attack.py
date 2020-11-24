## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 4501   # number of iterations to perform gradient descent
ABORT_EARLY = False       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-4    # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-6    # the initial constant c to pick as a first guess

class CarliniL2:
    def __init__(self, sess, model, noise, weight, batch_size, num_clips, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin =0, boxmax = 1):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """
        self.model = model
        self.image_size, self.num_channels, self.num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.num_clips = num_clips

        self.repeat = binary_search_steps >= 10

        shape = (self.batch_size,self.num_clips,self.image_size,self.image_size,self.num_channels)
        
        # the variable we're going to optimize over
        if noise is not None:
            modifier = tf.Variable(noise, dtype=tf.float32)
            weights = tf.constant(weight,dtype=tf.float32)
        else:
            modifier = tf.Variable(np.zeros(shape, dtype=np.float32) + 0.001)
            weights = tf.constant(np.ones(shape, dtype=np.float32))
        #modifier = tf.Variable(np.random.rand(self.batch_size,self.num_clips,self.image_size,self.image_size,self.num_channels), dtype=tf.float32)

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((self.batch_size,self.num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab0 = tf.placeholder(tf.int32, [self.batch_size])
        self.assign_tlab = tf.one_hot(self.assign_tlab0, self.num_labels)
        self.assign_const = tf.placeholder(tf.float32, [self.batch_size])

        self.tau = tf.placeholder(tf.float32, [1])
        
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        #self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus# * weights
        self.newimg = tf.minimum(tf.maximum(tf.add(modifier*10, self.timg)/255., 0.0), 1.0)

        # prediction BEFORE-SOFTMAX of the model
        self.output = model.predict(self.newimg)
        self.norm = tf.nn.softmax(self.output)
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg *255. - self.timg),[1, 2, 3, 4])
        #self.lidist = tf.reduce_sum(tf.maximum(0.0, tf.abs(modifier-self.tau)))
        #self.l2dist = tf.subtract(tf.image.total_variation(self.newimg[0]),tf.image.total_variation(self.timg[0]))
        #self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3,4])
        #self.tv1 = self.tv(modifier)

        # compute the probability of the label class versus the maximum other
        #self.real = tf.reduce_sum((self.tlab)*self.output,1)
        #self.other = tf.reduce_max((1 - self.tlab) * self.output - (self.tlab * 10000), 1)  # ,1
        self.real = tf.reduce_sum(tf.multiply(self.tlab, self.output), 1)
        self.other = tf.reduce_max(tf.subtract(1.,self.tlab)*self.output - tf.multiply(self.tlab,10000.),1)#,1

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = (self.other - self.real + self.CONFIDENCE)
            #loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.tlab))
            #loss1 = -tf.log(self.real + 1e-6)
        else:
            # if untargeted, optimize for making this class least likely.
            #loss1 = tf.maximum(0.0, self.real-self.other+self.CONFIDENCE)
            loss1 = -tf.log(1 - tf.multiply(self.tlab , self.norm))

        # sum up the losses
        self.loss2 = tf.reduce_sum(tf.multiply(self.const,self.l2dist))#+tf.reduce_sum(self.tv1)
        self.loss1 = tf.reduce_sum(loss1)#tf.maximum(0.0,(loss1-0.015)/0.2176+1)#self.const*
        self.loss = self.loss1 + self.loss2
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def tv(self,images):
        pixel_dif1 = images[:, 1:, :, :, :] - images[:, :-1, :, :, :]
        pixel_dif2 = images[:, :, 1:, :, :] - images[:, :, :-1, :, :]
        pixel_dif3 = images[:, :, :, 1:, :] - images[:, :, :, :-1, :]

        # Only sum for the last 3 axis.
        # This results in a 1-D tensor with the total variation for each image.
        sum_axis = [1, 2, 3, 4]

        tot_var = (math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) +
                   math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis) +
                   math_ops.reduce_sum(math_ops.abs(pixel_dif3), axis=sum_axis))
        return tot_var

    def attack(self, imgs, targets, true_lable):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('clip',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size], true_lable[i:i+self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, target_labs, true_lable):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        #imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const#
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestl1 = [1e10] * batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        o_con = np.ones(batch_size)*self.initial_const#

        #record_loss1 = np.zeros([self.BINARY_SEARCH_STEPS,self.MAX_ITERATIONS])
        #record_loss2 = np.zeros([self.BINARY_SEARCH_STEPS,self.MAX_ITERATIONS])
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print('outer_step: %s' % outer_step)
            print('o_bestl2: %s' %o_bestl2)
            print('CONST: %s' % CONST)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:self.num_clips]
            batchlab = target_labs[:batch_size]
            truelable = true_lable[:batch_size]
    
            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size
            ttau = 5

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                           self.assign_tlab0: batchlab,
                                           self.assign_const: CONST})
            
            prev = 1e6
            break_condition = False
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l1, lo2, l2s, scores, nimg, ireal, iother, norm = self.sess.run([self.train, self.loss, self.loss1, self.loss2,
                                                             self.l2dist, self.output, self.newimg,
                                                             self.real, self.other, self.norm], feed_dict={self.tau:[ttau]})

                    #record_loss1[outer_step][iteration] = l1
                    #record_loss2[outer_step][iteration] = l2

                if lo2 == 0 and self.const != 0:
                    ttau *= 0.9
                    print(ttau)
                if ttau==0:
                    break
                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration,(l,l1,lo2))


                # adjust the best result found so far
                for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
                    if self.TARGETED:
                        com_result = compare(sc, batchlab[e])
                    else:
                        com_result = compare(sc, truelable[e])
                    if l2 < bestl2[e] and com_result:#batchlab[e]
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if  l < o_bestl2[e]+o_bestl1[e]:#batchlab[e]   (l2 < o_bestl2[e] or l1 < o_bestl1[e]) and com_result
                        print('New best iteration: %s, l: %s, lable: %s, sorce: %s' % (
                            iteration, l, np.argmax(sc), norm[0][o_bestscore[e]]))
                        o_bestl2[e] = lo2.copy()
                        o_bestl1[e] = l1.copy()
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii
                        o_con = CONST.copy()
                    # check if we should abort search if we're getting nowhere.
                        if self.ABORT_EARLY and o_bestl2[e] < 0.001 and com_result:
                            break_condition = True


                if break_condition:
                    print('stop at %s' % iteration)
                    break


            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], batchlab[e]) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        #loss1_mean = np.mean(record_loss1)
        #loss1_std = np.std(record_loss1)
        #loss2_mean = np.mean(record_loss2)
        #loss2_std = np.std(record_loss2)

        #print('loss1_mean: %s' %loss1_mean)
        #print('loss1_std: %s' % loss1_std)
        #print('loss2_mean: %s' % loss2_mean)
        #print('loss2_std: %s' % loss2_std)


        # return the best solution found
        print('optimizer finish')
        print(o_bestl1)
        print("%s" %o_con)
        return o_bestattack
