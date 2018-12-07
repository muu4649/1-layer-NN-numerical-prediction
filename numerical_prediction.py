# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import tensorflow.contrib.slim as slim

def shuffle_batch(X,y,batch_size):#トレーニングデータをバッチ学習させる
   rnd_idx=np.random.permutation(len(X))
   n_batches=len(X)
   for batch_idx in np.array_split(rnd_idx,n_batches):
      X_batch,y_batch=X[batch_idx],y[batch_idx]
      yield X_batch,y_batch

# Parameters
learning_rate = 0.0005 # 学習率 高いとcostの収束が早まる0.001から下げていくのがbetter?
n_epochs = 200 # 学習全体をこのエポック数で区切り、区切りごとにcostを表示する

# Network Parameters
n_hidden1=     # 隠れ層1のユニットの数
n_hidden2=
n_inputs=          # 与える変数の数
n_outputs=

data=np.loadtxt("input.csv",delimiter=",",dtype="float")#入力変数
data1=np.genfromtxt("output.csv",delimiter=",",dtype="float",skip_header=1)#目的変数

X1=data[:,:]
Y1=data1[:,:]
X_train=X1[:,:]
y_train=Y1[:,:]
X_test=X1[:,:]
y_test=Y1[:,:]#トレーニングデータとテストデータの範囲指定める

y_train=np.reshape(y_train,(-1,1))
y_test=np.reshape(y_test,(-1,1))#目的変数を整える

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None), name="y")
sess = tf.Session()

with tf.name_scope("dnn"):#ニューラルネットワークの層定義
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1") 
    BN=tf.layers.BatchNormalization()
    x_BN=BN(hidden1)
    hidden2 = tf.layers.dense(x_BN, n_hidden2, activation=tf.nn.sigmoid, name="hidden2")       #活性関数　sigmoid
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")                         # new!
            # new!

with tf.name_scope("loss"):#損失
   # loss = tf.reduce_mean(tf.map_fn(tf.abs,y-logits),name="loss") 
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(y-logits),name="loss"))#MAE
     
with tf.name_scope("eval"):#評価
   # accuracy = tf.reduce_mean(tf.map_fn(tf.abs,y-logits), name="accuracy")
    accuracy = tf.reduce_mean(tf.reduce_mean(tf.square(y-logits),name="accuracy"))
     
with tf.name_scope('r2'):
    r2 = 1 - (tf.reduce_sum(tf.square(y - logits)) / tf.reduce_sum(tf.square(y - tf.reduce_mean(y))))#決定係数
  #  tf.summary.scalar('r2', r2)

with tf.name_scope('summary'):
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)#Tensorboardに必要なデータ出力

# Initializing the variables
init = tf.initialize_all_variables()


with tf.name_scope("train"):
   training_op= tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)#勾配法
   saver=tf.train.Saver()

tf.summary.scalar("loss",loss)#Tensorboard
tf.summary.scalar("accuracy",accuracy)#Tensorboard


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
           sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
           lossv=sess.run(loss,feed_dict={X:X_batch,y:y_batch})
           
           if epoch % 100 == 0:
              acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
              print(epoch, "Batch accuracy_loss:", acc_batch,lossv)
        print("--------------------------------------------------")

        sess.run(training_op, feed_dict={X: X_test, y: y_test})
        acc_valid, summary = sess.run([loss, merged], feed_dict={X: X_test, y: y_test})
        writer.add_summary(summary, epoch)
        print(epoch, "Test accuracy:",acc_valid)
    saver_path=saver.save(sess,"./NNmodel.ckpt")#ニューラルネットワークの重みとバイアスの記録





