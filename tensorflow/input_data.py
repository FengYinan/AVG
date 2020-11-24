# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time

def get_frames_data(filename, num_frames_per_clip=16):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  tem_arr = []
  s_index = 0
  filename = os.getcwd()+filename[1:]
  for parent, dirnames, filenames in os.walk(filename):
    if(len(filenames)<num_frames_per_clip):
      filenames = sorted(filenames)
      s_index = 0
      for i in range(s_index, len(filenames)):
        image_name = str(filename) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img_data = np.array(img)
        tem_arr.append(img_data)
    else:
      filenames = sorted(filenames)
      s_index = random.randint(0, len(filenames) - num_frames_per_clip)
      for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(filename) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img_data = np.array(img)
        tem_arr.append(img_data)

    valid_len = len(tem_arr)
    pad_len = num_frames_per_clip - valid_len
    if pad_len:
      for i in range(pad_len):
        tem_arr.append(img_data)

    ret_arr.append(tem_arr)
  return ret_arr, s_index



def read_clip_and_label(lines, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):

  read_dirnames = []
  data = []
  odata =[]

  img_datas = []
  original_datas = []


  batch_index = 0
  next_batch_start = -1

  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))

  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break

    dirname = lines[index]


    #print("Loading a video clip from {}...".format(dirname))

    tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)  #16 frames data

    img_datas = []
    original_datas = []
    if(len(tmp_data)!=0):
      for clip in tmp_data:
          for j in xrange(len(clip)):
            img = Image.fromarray(clip[j].astype(np.uint8))
            if(img.width>img.height):
              scale = float(crop_size)/float(img.height)
              img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
            else:
              scale = float(crop_size)/float(img.width)
              img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
            crop_x = int((img.shape[0] - crop_size)/2)
            crop_y = int((img.shape[1] - crop_size)/2)
            #img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] #- np_mean[j]
            box = (crop_y, crop_x, crop_y + crop_size, crop_x + crop_size)
            img = Image.fromarray(img.astype(np.uint8))
            nimg = img.crop(box)
            nimg = np.array(nimg)
            img_datas.append(nimg)
            original_datas.append(img)
          data.append(img_datas)
          odata.append(original_datas)
      read_dirnames.append(dirname)
      if len(img_datas) == num_frames_per_clip:
          batch_index = batch_index + 1
      else:
          read_dirnames.pop(-1)
          data.pop(-1)
          odata.pop(-1)

  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      odata.append(original_datas)


  np_arr_data = np.array(data).astype(np.float32)

  return np_arr_data, next_batch_start, read_dirnames, valid_len, box, scale


