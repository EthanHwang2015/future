#encoding=utf8
import argparse
import sys
reload(sys)
sys.setdefaultencoding("utf-8") 
sys.path.insert(0, '/home/healthai/tensorflow-1.0')

import tensorflow as tf
import pandas as pd

from loader import *
from main import Model
import pdb

def load_csv(fname, col_start=1, row_start=0, delimiter=",", dtype=dtypes.float32):
  #data = np.genfromtxt(fname, delimiter=delimiter)
  data = pd.read_csv(fname, sep=delimiter).values
  dates = data[:,0]
  for _ in range(col_start):
    data = np.delete(data, (0), axis=1)
  for _ in range(row_start):
    data = np.delete(data, (0), axis=0)
  # print(np.transpose(data))
  return dates, data


def load_predict_data(path, moving_window=128, columns=6):
  if not os.path.exists(path):
    print '{} is not exists'.format(path)
  def process_data(data):
    stock_set = np.zeros([0,moving_window,columns])
    for idx in range(data.shape[0] - moving_window+1):
      stock_set = np.concatenate((stock_set, np.expand_dims(data[range(idx,idx+(moving_window)),:], axis=0)), axis=0)

    return stock_set
  # read a directory of data
  #pdb.set_trace()
  stocks_set = np.zeros([0,moving_window,columns])
  dates, data = load_csv(path)
  ss = process_data(data)
  stocks_set = np.concatenate((stocks_set, ss), axis=0)
  stocks_set = stocks_set.astype(float)
  stocks_set_ = np.zeros(stocks_set.shape)
  for i in range(len(stocks_set)):
    min = stocks_set[i].min(axis=0)
    max = stocks_set[i].max(axis=0)
    stocks_set_[i] = (stocks_set[i] - min) / (max - min)
  stocks_set = stocks_set_

  results = {}
  lens = len(stocks_set)
  for i in range(1, lens+1):
    key = dates[len(dates)-i]
    value = stocks_set[len(stocks_set)-i]
    results[key] = value
  #return dates, stocks_set
  return results

if __name__ == '__main__':
  #key is date
  key = sys.argv[1]
  #test_datas = load_predict_data('./rb1710/rb1710(1小时).csv')
  test_datas = load_predict_data('./datas/rb1805_1hour.csv')
  imgs = np.expand_dims(test_datas[key], axis=0)
  print imgs.shape

  image = tf.placeholder(tf.float32, [None, 128, 6])
  label = tf.placeholder(tf.float32, [None, 3])
  dropout = tf.placeholder(tf.float32)
  model = Model(image, label, dropout=dropout)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  
  glob_step = 10001
  saver = tf.train.Saver()
  save_path = 'checkpoints/stocks_model.ckpt-{}'.format(glob_step)

  with tf.Session(config=config) as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path)
    pred = sess.run(model.prediction, {image: imgs, dropout:1.0})
    for i,p in enumerate(pred):
      print('{}: incr{}, decr{}, roll{}'.format(key, p[0], p[1], p[2]))
 

