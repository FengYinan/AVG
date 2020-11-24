import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
import PIL.Image as Image
import cv2
import input_data_googel
import torch

# https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
# https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/

class ImageData:

    def __init__(self, load_size, channels, augment_flag=False):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img


def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    img = Image.fromarray(images.astype(np.uint8))
    img.save(image_path)
    #return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def image_to_video(np_array,v_dir):
    video_dir = os.path.join(v_dir,'input.avi')
    fps = 30
    img_size = (np_array.shape[2],np_array.shape[3])

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # opencv3.0
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    #pre_frame = np_array[0][0] + 1

    for clip in np_array:
        for frame in clip:
            cv2_frame = frame[:, :, [2, 1, 0]].astype(np.uint8)
            #pre_frame = frame
            videoWriter.write(cv2_frame)

    videoWriter.release()
    return video_dir


if __name__ == '__main__':
    l = ['/home/google/hmdb51/shoot_gun/Shootingattherange_shoot_gun_u_nm_np1_ri_med_1']
    a,b,_,_,_,_ = input_data_googel.read_clip_and_label(l, 1, num_frames_per_clip=16, start_pos=-1)
    q = torch.from_numpy(a.transpose((0, 4, 1, 2, 3))).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).requires_grad_(False)
    q=q.cpu().detach().numpy().transpose((0, 2, 3, 4, 1))
    print(q.shape)
    print((q==a).all())
    x = image_to_video(q,'/home/google/test')
    for i in range(len(a)):
        for b in range(len(a[i])):
            save_images(q[i][b], 2, '/home/google/test/{}_{}.jpg'.format(i,b))


