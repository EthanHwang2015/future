#encoding=utf8
import argparse
import sys
reload(sys) 
sys.setdefaultencoding("utf-8")
#sys.path.insert(0, '/home/healthai/tensorflow-1.0')

import tensorflow as tf
import functools

from ops import *
from loader import *
import sklearn

def doublewrap(function):
  @functools.wraps(function)
  def decorator(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
      return function(args[0])
    else:
      return lambda wrapee: function(wrapee, *args, **kwargs)
  return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
  """
  A decorator for functions that define TensorFlow operations. The wrapped
  function will only be executed once. Subsequent calls to it will directly
  return the result so that operations are added to the graph only once.
  The operations added by the function live within a tf.variable_scope(). If
  this decorator is used with arguments, they will be forwarded to the
  variable scope. The scope name defaults to the name of the wrapped
  function.
  """
  attribute = '_cache_' + function.__name__
  name = scope or function.__name__
  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      with tf.variable_scope(name, *args, **kwargs):
        setattr(self, attribute, function(self))
    return getattr(self, attribute)
  return decorator

class Model:
  def __init__(self,
    image,
    label,
    dropout=0.5,
    learn_rate = 0.001,

    conv_size=9,
    conv_stride=1,
    ksize=2,
    pool_stride=2,
    filter_num=128,
    padding="SAME",
    class_weight = [5.0,5.0,1.0]):

    self.class_weight = class_weight

    self.image = image
    self.label = label
    self.dropout = dropout
    self.learn_rate = learn_rate

    self.conv_size = conv_size
    self.conv_stride = conv_stride
    self.ksize = ksize
    self.pool_stride = pool_stride
    self.padding = padding
    self.filter_num = filter_num

    self.prediction
    self.optimize
    self.accuracy

  @define_scope
  def prediction(self):
    with tf.variable_scope("model") as scope:
      #input image
      input_image = self.image

      layers = []

      # conv_1 [batch, ngf, 5] => [batch, 64, ngf]
      with tf.variable_scope("conv_1"):
        output = relu(conv1d(input_image, self.filter_num, name='conv_1', stddev=np.sqrt(2.0/self.filter_num)))
        layers.append(output)

      # conv_2 - conv_6
      layer_specs = [
        (self.filter_num * 2, self.dropout),  # conv_2: [batch, 64, ngf] => [batch, 32, ngf * 2]
        (self.filter_num * 4, self.dropout),  # conv_3: [batch, 32, ngf * 2] => [batch, 16, ngf * 4]
        (self.filter_num * 8, self.dropout),  # conv_4: [batch, 16, ngf * 4] => [batch, 8, ngf * 8]
        (self.filter_num * 8, self.dropout),  # conv_5: [batch, 8, ngf * 8] => [batch, 4, ngf * 8]
        (self.filter_num * 8, self.dropout)  # conv_6: [batch, 4, ngf * 8] => [batch, 2, ngf * 8]
      ]

      # adding layers
      for _, (out_channels, dropout) in enumerate(layer_specs):
        with tf.variable_scope("conv_%d" % (len(layers) + 1)):
          rectified = lrelu(layers[-1], 0.2)

          # [batch, in_width, in_channels] => [batch, in_width/2, out_channels]
          convolved = conv1d(rectified, out_channels, stddev=np.sqrt(2.0/out_channels))

          # batchnormalize convolved
          output = batchnorm(convolved, is_2d=False)

          # dropout
          if dropout > 0.0:
            output = tf.nn.dropout(output, keep_prob=dropout)

          layers.append(output)

      #fc1
      h_fc1 = relu(fully_connected(layers[-1], 256, name='fc1'))

      #dropout
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self.dropout)

      #fc2
      result = tf.nn.softmax(fully_connected(h_fc1_drop, 3, name='fc2'))

      return result

  def focal_loss2(self, logits, labels, alpha=0.25, gamma=2):
    sigmoid_p = tf.nn.sigmoid(logits)
    zeros = tf.zeros_like(sigmoid_p)
    pos_p_sub = tf.where(labels > zeros, labels - sigmoid_p, zeros)

    neg_p_sub = tf.where(labels > zeros, zeros, sigmoid_p)

    ce = -alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
         - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(ce)

  def focal_loss(self, logits, labels, alpha=0.25, gamma=2):
    #ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits = logits)
    probs = tf.nn.softmax(logits)

    ce = tf.multiply(labels, -tf.log(probs))

    #alpha should be set up by data distributiopn
    alpha = tf.convert_to_tensor(self.class_weight, dtype=probs.dtype)

    probs_t = tf.where(labels > 0, probs, 1.0 - probs)

    weight = tf.pow(tf.subtract(1.0, probs_t), gamma)

    fl = tf.multiply(alpha, tf.multiply(weight, ce))

    loss = tf.reduce_mean(fl)

    return loss


  @define_scope
  def optimize(self):
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
#      logits=self.prediction))
#    return tf.train.AdamOptimizer(self.learn_rate).minimize(cross_entropy)
    #return tf.train.MomentumOptimizer(self.learn_rate, 0.9).minimize(cross_entropy)
    loss = self.focal_loss(self.prediction, self.label)

    return tf.train.AdamOptimizer(self.learn_rate).minimize(loss)

  @define_scope
  def accuracy(self):
    correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # @define_scope
  # def optimize(self):
  #   with tf.name_scope("loss"):
  #     loss = tf.reduce_mean(tf.abs(self.p_loss))
  #   tvars = tf.trainable_variables()
  #   optim = tf.train.AdamOptimizer(0.0001)
  #   grads_and_vars = optim.compute_gradients(loss, var_list=tvars)
  #   print(grads_and_vars)
  #   train = optim.apply_gradients(grads_and_vars)

  @define_scope
  def p_loss(self):
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
#      logits=self.prediction))
#    return cross_entropy
    loss = self.focal_loss(self.prediction, self.label)
    return loss


  @define_scope
  def confusion(self):
    confu = tf.confusion_matrix(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1),
        num_classes=3,
        name = 'batch_confusion')
    return confu

def cal_kappa(y, pred, nclasses=3):
    nclasses = max(y) - min(y) + 1 
    o = np.zeros([nclasses,nclasses])
    w =  np.zeros([nclasses,nclasses])
    y_hist = np.zeros(nclasses)
    pred_hist = np.zeros(nclasses)
    for i in xrange(nclasses):
        for j in xrange(nclasses):
            w[i,j] = (i-j)**2
    for i in xrange(y.shape[0]):
        o[int(round(y[i])), int(round(pred[i]))] += 1
        y_hist[int(round(y[i]))] += 1
        pred_hist[int(round(pred[i]))] += 1
    e = np.outer(y_hist, pred_hist)
    rescale = np.sum(e) / np.sum(o)
    return 1 - rescale * np.sum(o * w) / np.sum(e * w)

def main():
  # Import data
  #db = load_stock_data("data/")
  name = 'i9888_60min'
  db, class_weight = load_stock_data("data20190508/{}.csv".format(name), rate=1.01)
  save_path = 'checkpoints/'
  model_name = '{}_best_validation.ckpt'.format(name)

  # Construct graph
  image = tf.placeholder(tf.float32, [None, 128, 6])
  label = tf.placeholder(tf.float32, [None, 3])
  dropout = tf.placeholder(tf.float32)

  learn_rate = 1e-4

  model = Model(image, label, dropout=dropout, learn_rate = learn_rate, class_weight=class_weight)

  # Saver
  saver = tf.train.Saver()

  improved_str = ''
  best_loss = 10000.0
  required_improved = 500
  last_improved = 0

  # Session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500000):
      images, labels = db.train.next_batch(128)
      print "step {}".format(i)
      if i % 100 == 0:
#if i > 0:
        #train accuracy
        accuracy = sess.run(model.accuracy, {image: images, label: labels, dropout: 1.0})
        loss = sess.run(model.p_loss, {image: images, label: labels, dropout: 1.0})
        pred = sess.run(model.prediction, {image: images, label: labels, dropout: 1.0})
        print('step %d, train: accuracy %g, loss %g, kappa %g' % (i, accuracy, loss, cal_kappa(np.argmax(labels, axis=1), np.argmax(pred, axis=1))))
        confusion = sess.run(model.confusion, {image: images, label: labels, dropout: 1.0})
        print('confusion\n {}\n'.format(confusion))

        #valid accuracy 
        images_eval, labels_eval = db.test.next_batch(1000,False)
        eval_accuracy = sess.run(model.accuracy, {image: images_eval, label: labels_eval, dropout: 1.0})
        eval_loss = sess.run(model.p_loss, {image: images_eval, label: labels_eval, dropout: 1.0})
        eval_pred = sess.run(model.prediction, {image: images_eval, label: labels_eval, dropout: 1.0})

        f = 0.5
        pred0 = np.argmax(eval_pred, axis=1)
        true0 = np.argmax(labels_eval, axis=1)
        rep = sklearn.metrics.classification_report(true0, pred0)
        print("{}, {}".format(f, rep))


        f = 0.7
        pred0 = np.argmax(np.where(eval_pred > f, 1, 0), axis=1)
        true0 = np.argmax(labels_eval, axis=1)
        rep = sklearn.metrics.classification_report(true0, pred0)

        print("{}, {}".format(f, rep))

        f = 0.8
        pred0 = np.argmax(np.where(eval_pred > f, 1, 0), axis=1)
        true0 = np.argmax(labels_eval, axis=1)
        rep = sklearn.metrics.classification_report(true0, pred0)

        print("{}, {}".format(f, rep))


        f = 0.9
        pred0 = np.argmax(np.where(eval_pred > f, 1, 0), axis=1)
        true0 = np.argmax(labels_eval, axis=1)
        rep = sklearn.metrics.classification_report(true0, pred0)

        print("{}, {}".format(f, rep))




        if eval_loss < best_loss:
#if eval_loss < 0:
          best_loss = eval_loss
          improved_str = '*'
          last_improved = i

          if not os.path.exists(save_path):
            os.makedirs(save_path)
          save_path_full = os.path.join(save_path, model_name)
          saver.save(sess, save_path_full)
     
        else:
          improved_str=''
        print('step %d, valid: accuracy %g, loss %g, kappa %g %s' % (i, eval_accuracy, eval_loss, cal_kappa(np.argmax(labels_eval, axis=1), np.argmax(eval_pred, axis=1)), improved_str))
        confusion = sess.run(model.confusion, {image: images_eval, label: labels_eval, dropout: 1.0})
        print('confusion\n {}\n'.format(confusion))


      sess.run(model.optimize, {image: images, label: labels, dropout: 0.5})

      if (i-last_improved) > required_improved:
        if model.learn_rate > 1e-5:
          model.learn_rate = model.learn_rate * 0.1
          print('adjust learn rate to : %g' % (model.learn_rate))
          last_improved = i
        else:
          break
       #exit(0)

    images_eval, labels_eval = db.test.next_batch(1000,False)
    accuracy = sess.run(model.accuracy, {image: images_eval, label: labels_eval, dropout: 1.0})
    print('final accuracy on testing set: %g' % (accuracy))
  print("finished")


if __name__ == '__main__':
  main()
