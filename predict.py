#encoding=utf8
import argparse
import sys
reload(sys)
sys.setdefaultencoding("utf-8") 
import sklearn
#sys.path.insert(0, '/home/healthai/tensorflow-1.0')

import tensorflow as tf
import pandas as pd

from loader import *
from main import Model
import pdb

def load_csv(fname, col_start=0, row_start=0, delimiter=",", dtype=dtypes.float32):
  data = np.genfromtxt(fname, delimiter=delimiter, dtype=str)

#data = pd.read_csv(fname, sep=delimiter).values
  dates = data[:,0]
  values = data[:,1:]
  return dates, values.astype(np.float)


def load_predict_data(path, moving_window=128, columns=6, rate=1.01):
  if not os.path.exists(path):
    print '{} is not exists'.format(path)
  def process_data(data):
    stock_set = np.zeros([0,moving_window,columns])
    label_set = np.zeros([0,3])
    for idx in range(data.shape[0] - moving_window-2):
      stock_set = np.concatenate((stock_set, np.expand_dims(data[range(idx,idx+(moving_window)),:], axis=0)), axis=0)
      if data[idx+(moving_window+2),3] >= data[idx+(moving_window),3]*rate or\
          data[idx+(moving_window+1),3] >= data[idx+(moving_window),3]*rate :
        lbl = [[1.0, 0.0, 0.0]]
#     elif data[idx+(moving_window+3),3]*rate <= data[idx+(moving_window),3] or\
      elif data[idx+(moving_window+2),3]*rate <= data[idx+(moving_window),3] or\
          data[idx+(moving_window+1),3]*rate <= data[idx+(moving_window),3] :
        lbl = [[0.0, 1.0, 0.0]]
      else:
        lbl = [[0.0, 0.0, 1.0]]
      label_set = np.concatenate((label_set, lbl), axis=0)
 

    return stock_set, label_set
  # read a directory of data
  #pdb.set_trace()
  stocks_set = np.zeros([0,moving_window,columns])
  labels_set = np.zeros([0,3])

  dates, data = load_csv(path)
  ss, ls = process_data(data)
  stocks_set = np.concatenate((stocks_set, ss), axis=0)
  labels_set = np.concatenate((labels_set, ls), axis=0)

  stocks_set = stocks_set.astype(float)
  stocks_set_ = np.zeros(stocks_set.shape)
  for i in range(len(stocks_set)):
    min = stocks_set[i].min(axis=0)
    max = stocks_set[i].max(axis=0)
    stocks_set_[i] = (stocks_set[i] - min) / (max - min)
  stocks_set = stocks_set_

  return dates[moving_window-1:], stocks_set, labels_set

if __name__ == '__main__':
  #key is date
#  key = sys.argv[1]
  #test_datas = load_predict_data('./rb1710/rb1710(1小时).csv')
  path = './data20190623/test_i9888_60min.csv'
  dates, imgs, labels  = load_predict_data('./data20190623/test_i9888_60min.csv')
  #imgs = np.expand_dims(test_datas[key], axis=0)
  print dates.shape,imgs.shape
#  pdb.set_trace()

  image = tf.placeholder(tf.float32, [None, 128, 6])
  label = tf.placeholder(tf.float32, [None, 3])
  dropout = tf.placeholder(tf.float32)
  model = Model(image, label, dropout=dropout)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
#exit(0)
  
  glob_step = 'data-00000-of-00001'#10001
  saver = tf.train.Saver()
#save_path = 'models/i9888_60min_model/i9888_60min_best_validation.ckpt.{}'.format(glob_step)
  save_path = 'checkpoints/i9888_60min_best_validation.ckpt.{}'.format(glob_step)

  with tf.Session(config=config) as sess:
    #sess.run(tf.global_variables_initializer())
#saver.restore(sess, save_path)
    saver = tf.train.import_meta_graph("./models/i9888_60min_model/i9888_60min_best_validation.ckpt.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./models/i9888_60min_model/"))
    pred = sess.run(model.prediction, {image: imgs, label:labels, dropout:1.0})


    pred0 = np.argmax(pred, axis=1)
    true0 = np.argmax(labels, axis=1)
    rep = sklearn.metrics.classification_report(true0, pred0)
    print("{}".format(rep))


    for i,p in enumerate(pred):
      if p[0] > 0.5 or p[1] > 0.5:
        #if np.argmax(p) > 0.9:
        print('{}: incr{}, decr{}, roll{}, predict:{}, true:{}'.format(dates[i], p[0], p[1], p[2], np.argmax(p),np.argmax(labels[i])))
 

