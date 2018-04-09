#-*-coding:utf-8-*-
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys
import os
import random
import tensorflow as tf
import logging
from logging.config import fileConfig
import time
sys.path.append("./")
sys.path.append("./data/")
import data_preprocess
from util import Eval,Merge
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("LSTM_1")


class Config(object):
    '''
    默认配置
    '''
    learning_rate=0.05
    batch_size=200
    sent_len=80    # 问句长度
    embedding_dim=100    #词向量维度
    hidden_dim=100
    train_dir='./data/train_out_%s.txt'
    dev_dir='./data/dev_out.txt'
    test_dir='./data/test.txt'
    model_dir='./save_model/model_%s/r_net_model_%s.ckpt'
    if not os.path.exists('./save_model/model_%s_'):
        os.makedirs('./save_model/model_%s_')
    use_cpu_num=8
    keep_dropout=0.7
    summary_write_dir="./tmp/r_net.log"
    epoch=10
    lambda1=0.01
    model_mode='bilstm_attention_crf' #模型选择：bilstm bilstm+crf bilstm+attention bilstm+attention+crf

config=Config()
tf.app.flags.DEFINE_float("lambda1", config.lambda1, "l2学习率")
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_float("keep_dropout", config.keep_dropout, "dropout")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("sent_len", config.sent_len, "句子长度")
tf.app.flags.DEFINE_integer("embedding_dim", config.embedding_dim, "词嵌入维度.")
tf.app.flags.DEFINE_integer("hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("epoch", config.epoch, "epoch次数")
tf.app.flags.DEFINE_string("summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_string("mod", "infer", "默认为训练") # true for prediction
tf.app.flags.DEFINE_string('model_mode',config.model_mode,'模型类型')
FLAGS = tf.app.flags.FLAGS




class Bilstm(object):

    def __init__(self,num_class,hidden_dim,seq_len,seq_dim,mode,batch_size,embedding_dim,vocab_num,crf_mode,loss_weight_mode,iter_num):
        self.num_class=num_class
        self.hidden_dim=hidden_dim
        self.seq_len=seq_len
        self.seq_dim=seq_dim
        self.iter_num=iter_num
        self.mode=mode
        self.batch_size=batch_size
        self.seq_vec=tf.placeholder(shape=(None,),dtype=tf.int32)
        self.loss_weight_mode=loss_weight_mode
        self.embedding_dim=embedding_dim
        self.embedding=tf.Variable(tf.random_uniform(shape=(vocab_num,embedding_dim),minval=-1.0,maxval=1.0,dtype=tf.float32))
        self.X_sent=tf.placeholder(shape=(None,self.seq_len),dtype=tf.int32)
        self.input_emb=tf.nn.embedding_lookup(self.embedding,self.X_sent)
        self.Y = tf.placeholder(shape=(None,self.seq_len), dtype=tf.int32)
        X_=tf.transpose(self.input_emb,[1,0,2])
        X_= tf.unstack(X_,self.seq_len,0)
        self.crf_mode = crf_mode
        self.lstm_input=X_
        with tf.device("/cpu:3"):#3
            cell_fw = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)
            cell_bw = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                state_is_tuple=False)

            (lstm_out, states, _) = tf.contrib.rnn.static_bidirectional_rnn(
                cell_fw, cell_bw, self.lstm_input, dtype=tf.float32,
                sequence_length=self.seq_vec)

        with tf.device("/cpu:4"):#4
            self.lstm_output=tf.nn.dropout(lstm_out,keep_prob=1.0)
            lstm_out=tf.stack(self.lstm_output,0)
            lstm_out=tf.transpose(lstm_out,[1,0,2])#[batch_size,seq_len,2*hidden_dim]

            if self.mode == "bilstm":
                lstm_w = tf.Variable(
                    tf.random_normal([self.hidden_dim * 2, self.num_class], stddev=1, seed=1, dtype=tf.float32))
                lstm_b = tf.Variable(tf.random_normal([self.num_class], dtype=tf.float32), name="b")
                tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(lstm_w))
                self.logit = tf.add(tf.einsum("ijk,kl->ijl", lstm_out, lstm_w), lstm_b)
                self.lstm_prediction = tf.nn.softmax(self.logit)
                YY = tf.one_hot(self.Y, self.num_class, 1, 0)
                #YY = tf.reshape(YY, (self.seq_len * self.num_class))
                #lstm_prediction = tf.reshape(self.lstm_prediction, (self.seq_len * self.num_class))
                YY = tf.reshape(YY, (1,-1))
                lstm_prediction = tf.reshape(self.lstm_prediction, (1,-1))
                #self.loss_op = tf.losses.mean_squared_error(YY, lstm_prediction)
                #self.loss_op=tf.losses.softmax_cross_entropy(YY,lstm_prediction)
                self.loss_op=self.loss_entory(YY,lstm_prediction)


                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.4).minimize(self.loss_op)
                self.loss_summary = tf.summary.scalar("loss", self.loss_op)

            elif self.mode == "bilstm_attention":
                lstm_w = tf.Variable(
                    tf.random_normal([self.hidden_dim * 4, self.num_class], stddev=1, seed=1, dtype=tf.float32))
                lstm_b = tf.Variable(tf.random_normal([self.num_class], dtype=tf.float32), name="b")
                lstm_out = self.attention(lstm_out)
                tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(lstm_w))
                self.logit = tf.add(tf.einsum("ijk,kl->ijl", lstm_out, lstm_w), lstm_b)
                self.lstm_prediction = tf.nn.softmax(self.logit)
                YY1 = tf.one_hot(self.Y, self.num_class, 1, 0,2)
                YY = tf.reshape(YY1, (1, -1))
                lstm_prediction = tf.reshape(self.lstm_prediction, (1, -1))
                self.loss_op=tf.losses.softmax_cross_entropy(YY1,self.lstm_prediction)
                #self.loss_op=self.loss_entory(YY,lstm_prediction)
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.4).minimize(self.loss_op)
                self.loss_summary = tf.summary.scalar("loss", self.loss_op)

            elif self.mode == "bilstm_crf":
                lstm_w = tf.Variable(
                    tf.random_normal([self.hidden_dim * 2, self.num_class], stddev=1, seed=1, dtype=tf.float32))
                lstm_b = tf.Variable(tf.random_normal([self.num_class], dtype=tf.float32), name="b")
                #lstm_out = self.attention(lstm_out)
                tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(lstm_w))
                logit = tf.add(tf.einsum("ijk,kl->ijl", lstm_out, lstm_w), lstm_b)
                #logit = tf.add(tf.einsum("ijk,kl->ijl", lstm_out, lstm_w), lstm_b)
                softmax_prediction = tf.nn.softmax(logit)

                if self.crf_mode=="train":
                    self.loss_op,self.max_score,self.max_score_pre=self.crf(softmax_lstm=softmax_prediction,crf_mode=self.crf_mode)
                    self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.4).minimize(self.loss_op)
                    self.loss_summary = tf.summary.scalar("loss", self.loss_op)
                elif self.crf_mode=="decode":
                    self.max_score, self.max_score_pre=self.crf(softmax_lstm=softmax_prediction,crf_mode=self.crf_mode)

            elif self.mode=="bilstm_attention_crf":
                lstm_w = tf.Variable(
                    tf.random_normal([self.hidden_dim * 4, self.num_class], stddev=1, seed=1, dtype=tf.float32))
                lstm_b = tf.Variable(tf.random_normal([self.num_class], dtype=tf.float32), name="b")
                lstm_out = self.attention(lstm_out)
                lstm_out=tf.nn.dropout(lstm_out,0.5)
                tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(lstm_w))
                logit = tf.add(tf.einsum("ijk,kl->ijl", lstm_out, lstm_w), lstm_b)
                #label = tf.one_hot(self.Y, self.num_class, 1, 0,2)

                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    logit, self.Y, self.seq_vec)
                self.logit=logit
                self.trans_params = trans_params  # need to evaluate it for decoding
                self.loss_op = tf.reduce_mean(-log_likelihood)
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.4).minimize(self.loss_op)

            elif self.mode=="bilstm_self_attention_crf":
                lstm_w = tf.Variable(
                    tf.random_normal([self.hidden_dim * 2, self.num_class], stddev=1, seed=1, dtype=tf.float32))
                lstm_b = tf.Variable(tf.random_normal([self.num_class], dtype=tf.float32), name="b")
                lstm_out=self.self_lstm_attention(lstm_out)
                lstm_out=tf.nn.dropout(lstm_out,0.8)
                tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(lstm_w))
                logit = tf.add(tf.einsum("ijk,kl->ijl", lstm_out, lstm_w), lstm_b)
                #label = tf.one_hot(self.Y, self.num_class, 1, 0,2)

                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    logit, self.Y, self.seq_vec)
                self.logit=logit
                self.trans_params = trans_params  # need to evaluate it for decoding
                self.loss_op = tf.reduce_mean(-log_likelihood)
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.4).minimize(self.loss_op)

    def loss_entory(self,y,y_):
        '''
        交叉熵损失函数,y为真实标注，y_为预测类别
        :param y:
        :param y_:
        :return:
        '''
        y=tf.cast(y,tf.float32)
        #y_=y*y_
        if self.loss_weight_mode:
            y=tf.gather(y[0],self.loss_weight)
            y_=tf.gather(y_[0],self.loss_weight)
        loss = y * tf.log(y_) + (1 - y) * tf.log(1 - y_)
        #loss = y*tf.log(y_)
        loss = -tf.reduce_mean(loss)

        #loss=tf.reduce_mean(tf.abs(y_-y))

        return loss

    def attention(self,lstm_outs):
        '''
        输入lstm的输出组，进行attention处理
        :param lstm_outs:
        :return:
        '''

        '''
                w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.seq_len)))
        b_h=tf.Variable(tf.random_normal(shape=(self.seq_len,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        G=tf.nn.softmax(tf.nn.tanh(tf.add(logit,b_h)))#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)
        '''
        w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,2*self.hidden_dim)))
        b_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        logit=tf.nn.tanh(tf.add(logit,b_h))
        logit=tf.tanh(tf.einsum("ijk,ilk->ijl",logit,lstm_outs))
        G=tf.nn.softmax(logit)#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)


        # 注意力得到的logit与lstm_outs进行链接

        outs=tf.concat((logit_,lstm_outs),2)#outs.shape=[None,seq_len,4*hidden_dim]
        return outs

    def self_lstm_attention_ops(self,lstm_out_t,lstm_outs):
        '''

        :return:
        '''
        w=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,2*self.hidden_dim))) #lstm_out_t 参数
        g=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,2*self.hidden_dim))) #lstm_outs 参数
        lstm_out_t=tf.reshape(lstm_out_t,[-1,1,2*self.hidden_dim])

        v=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,1)))
        with tf.variable_scope('self_attention',reuse=True):
            lstm_out_t_=tf.einsum('ijk,kl->ijl',lstm_out_t,w)
            lstm_outs_=tf.einsum('ijk,kl->ijl',lstm_outs,g)
            gg=tf.tanh(lstm_out_t_+lstm_outs_)
            gg_=tf.einsum('ijk,kl->ijl',gg,v)
            gg_soft=tf.nn.softmax(gg_,1)
            a=tf.einsum('ijk,ijl->ikl',lstm_outs,gg_soft)
            a=tf.reshape(a,[-1,2*self.hidden_dim])
            return a

    def self_lstm_attention(self,lstm_outs):
        '''
        对lstm输出再做一层 attention_lstm
        :param lstm_outs:
        :return:
        '''

        lstm_cell=tf.contrib.rnn.LSTMCell(2*self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                state_is_tuple=True)
        lstm_outs_list=tf.unstack(lstm_outs,self.seq_len,1)
        init_state=tf.zeros_like(lstm_outs_list[0])
        states=[(init_state,init_state)]
        H=[]
        w=tf.Variable(tf.random_uniform(shape=(4*self.hidden_dim,4*self.hidden_dim)))
        with tf.variable_scope('lstm_attention'):
            for i in range(self.seq_len):
                if i>0:
                    tf.get_variable_scope().reuse_variables()
                lstm_outs_t=lstm_outs_list[i]
                a=self.self_lstm_attention_ops(lstm_outs_t,lstm_outs) #attention的值

                new_input=tf.concat((lstm_outs_t,a),1)

                new_input_=tf.sigmoid(tf.matmul(new_input,w))*new_input

                h,state=lstm_cell(new_input_,states[-1])
                H.append(h)
                states.append(state)
        H=tf.stack(H)
        H=tf.transpose(H,[1,0,2])
        return H

    def crf_acc(self,pre_label,real_label,rel_len):
        """
        
        :param best_path: 
        :param path: 
        :return: 
        """
        real_labels_all = []
        for label, r_len in zip(real_label, rel_len):
            real_labels_all.extend(label[:r_len])

        verbit_seq_all=[]
        for seq,r_len in zip(pre_label,rel_len):
            verbit_seq_all.extend(seq[:r_len])

        best_path=verbit_seq_all
        path=real_labels_all
        #ss = sum([1 for i, j in zip(best_path, path) if int(i) == int(j)])
        #length = sum([1 for i in path if int(i) != 0])
        if len(best_path)!=len(path):
            print("error")
        else:

            ss = sum([1 for i, j in zip(best_path, path) if int(i) == int(j) and int(i) != 0])
            length = sum([1 for i, j in zip(best_path, path) if int(i) != 0 or int(j) != 0])
            acc = (float(ss) / float(length))
            return acc

    def Verbit(self,logits,batch_size,trans_params,sequence_lengths):

        viterbi_sequences = []
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit_ = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit_, trans_params)
            viterbi_sequences += [viterbi_seq]
        viterbi_sequences = viterbi_sequences
        return viterbi_sequences

    def train(self,dd):
        '''
        训练模块
        :param dd:
        :return:
        '''
        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=16,
                                intra_op_parallelism_threads=16,
                                log_device_placement=False)
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            if os.path.exists('./save_model/%s.ckpt.meta'%self.mode):
                saver.restore(sess,"./save_model/%s.ckpt"%self.mode)
                _logger.info('Load Model from %s file!'%self.mode)
            else:
                _logger.info('Initializer model params')
                sess.run(tf.global_variables_initializer())

            train_sent,train_sent_len,train_label=dd.next_batch()
            # dev_sent,dev_entity,dev_label,dev_loss_weight,dev_seq_vec=dd.get_dev()
            init_train_loss = 999.99
            init_dev_loss = 999.99
            init_train_acc = 0.0
            init_dev_acc = 0.0
            for i in range(FLAGS.epoch):
                for j in range(self.iter_num):
                    _logger.info('This is %s epoch,iter %s'%(i,j))
                    sent, sent_len, label,_ = dd.next_batch()

                    train_loss, _, logit, trans_params = sess.run(
                        [self.loss_op, self.optimizer, self.logit, self.trans_params], feed_dict={
                            self.X_sent: sent,
                            self.Y: label,
                            self.seq_vec: sent_len,
                        })

                    # dev_loss,dev_logit,dev_trans_params=sess.run([self.loss_op,self.logit,self.trans_params],feed_dict={self.X_sent:dev_sent,
                    #                                                           self.Y:dev_label,
                    #                                                            self.seq_vec:dev_seq_vec,
                    #                                                                self.loss_weight:dev_loss_weight})


                    verbit_seq = self.Verbit(logits=logit, batch_size=sent.shape[0], trans_params=trans_params,
                                             sequence_lengths=sent_len)
                    # dev_verbit_seq=self.Verbit(logits=dev_logit,batch_size=dev_sent.shape[0],trans_params=dev_trans_params,sequence_lengths=dev_seq_vec)
                    eval=Eval(label,verbit_seq)

                    crf_acc_train = self.crf_acc(verbit_seq, label, sent_len)
                    # crf_acc_dev=self.crf_acc(dev_verbit_seq,dev_label,dev_seq_vec)

                    _logger.info("训练误差:%s,训练准确率:%s"%(train_loss, crf_acc_train))
                    # _logger.info("验证误差,验证准确率",dev_loss,crf_acc_dev)
                    if train_loss < init_train_loss:
                        _logger.info("save model")
                        init_train_loss = train_loss
                        saver.save(sess, "./save_model/%s.ckpt"%self.mode)
                    print('\n')

    def _train(self,dd):
        '''
                训练模块
                :param dd:
                :return:
                '''
        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=2,
                                intra_op_parallelism_threads=2,
                                log_device_placement=False)
        saver = tf.train.Saver()
        id2word=dd.id2word
        id2label=dd.id2label

        wf=open('./out.txt','w')

        with tf.Session(config=config) as sess:
            if os.path.exists('./save_model/%s.ckpt.meta' % self.mode):
                saver.restore(sess, "./save_model/%s.ckpt" % self.mode)
                _logger.info('Load Model from %s file!' % self.mode)
            else:
                _logger.info('Initializer model params')
                sess.run(tf.global_variables_initializer())

            # sent, sent_len, label = dd.sent, dd.sent_len, dd.label
            # dev_sent,dev_entity,dev_label,dev_loss_weight,dev_seq_vec=dd.get_dev()
            init_train_loss = 999.99
            init_dev_loss = 999.99
            init_train_acc = 0.0
            init_dev_acc = 0.0
            mm=Merge()
            for _ in range(self.iter_num):
                sent, sent_len, label,_ = dd.next_batch()

                train_loss, logit, trans_params = sess.run(
                    [self.loss_op, self.logit, self.trans_params], feed_dict={
                        self.X_sent: sent,
                        self.Y: label,
                        self.seq_vec: sent_len,
                    })
                _logger.info(train_loss)
                verbit_seq = self.Verbit(logits=logit, batch_size=sent.shape[0], trans_params=trans_params,
                                         sequence_lengths=sent_len)


                for e, e1,e2 in zip(label, verbit_seq,sent):
                    e2=e2[:len(e1)]
                    e=e[:len(e1)]
                    e2=[id2word[ele] for ele in e2]
                    e=[id2label[str(ele)] for ele in e]
                    e1=[id2label[str(ele)] for ele in e1]
                    real_word=mm.merge_word(e2,e)
                    pre_word=mm.merge_word(e2,e1)
                    # e = " ".join([id2label[str(e_)] for e_ in e])
                    # e1 = ' '.join([id2label[str(e1_)] for e1_ in e1])

                    wf.write(real_word)
                    wf.write('\n')
                    wf.write(pre_word)
                    wf.write('\n')
                    wf.write('\n')

                eval = Eval(label, verbit_seq)
                for i in range(self.num_class):
                    print('id:%s,precision:%s,recall:%s' % (i, eval.precision(i), eval.recall(i)))
                    print('\n')

    def merge_word(self,words,labels):
        '''

        :param word:
        :param label:
        :return:
        '''
        words=list(words)
        labels=list(labels)
        new_words = []
        length = len(labels)
        f = 0
        for i in range(length):
            i = f
            if i <= length - 1:
                if labels[i] == '0' or labels[i].endswith('E'):
                    new_words.append(words[i])
                    f += 1
                elif labels[i].endswith('B'):
                    ss = []
                    ss.append(words[i])
                    for j in range(i + 1, length):
                        if labels[j][0] == labels[i][0]:
                            if labels[j].endswith('M') or labels[j].endswith('E'):
                                ss.append(words[j])
                            else:
                                break
                        else:
                            break
                    f += len(ss)
                    sss = ''.join(ss)
                    new_words.append(sss)
                elif labels[i].endswith('M'):
                    ss = []
                    ss.append(words[i])
                    for j in range(i + 1, length):
                        if labels[j][0] == labels[i][0]:
                            if labels[j].endswith('M') or labels[j].endswith('E'):
                                ss.append(words[j])
                            else:
                                break
                        else:
                            break
                    f += len(ss)
                    sss = ''.join(ss)
                    new_words.append(sss)
                else:
                    new_words.append(words[i])
                    f += 1
        return " ".join(new_words)

    def infer(self,sent_vec,sent_len):
        '''
        crf解码预测
        :param sent_array:
        :return:
        '''

        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=16,
                                intra_op_parallelism_threads=16,
                                log_device_placement=False)
        saver = tf.train.Saver()
        _logger.info('')
        with tf.Session(config=config) as sess:
            print("load model!")
            saver.restore(sess, "./save_model/%s.ckpt" % self.mode)

            logit, trans_params = sess.run([self.logit, self.trans_params],
                                           feed_dict={self.X_sent: sent_vec,
                                                      self.seq_vec: sent_len,
                                                      })
            verbit_seq = self.Verbit(logits=logit, batch_size=sent_vec.shape[0], trans_params=trans_params,
                                     sequence_lengths=sent_len)

            return verbit_seq






if __name__ == '__main__':
    start_time=time.time()
    with tf.device("/cpu:0"):
        _logger.info("load data")
        dd = data_preprocess.Entity_Extration_Data(train_path="./data/data1.txt", test_path="./data/test.txt",
                            dev_path="./data/dev.txt", batch_size=FLAGS.batch_size ,sent_len=FLAGS.sent_len, flag="train_new")
        num_class=len(dd.label_vocab)
        vocab_num=len(dd.vocab)
        id2label=dd.id2label
        iter_num=int(int(dd.file_size)/int(FLAGS.batch_size))
        nn_model = Bilstm(hidden_dim=FLAGS.hidden_dim,
                      seq_len=FLAGS.sent_len, seq_dim=FLAGS.embedding_dim,
                      num_class=num_class,
                      mode=FLAGS.model_mode,batch_size=FLAGS.batch_size,crf_mode='train',loss_weight_mode='train',vocab_num=vocab_num
                          ,embedding_dim=FLAGS.embedding_dim,iter_num=iter_num)

        _logger.info("load data finish")
        if FLAGS.mod=='train':
            nn_model.train(dd)
        elif FLAGS.mod=='infer':
            while True:
                sent=input('input:')
                sent_vec,sent_len=dd.get_sent(sent)
                seq=nn_model.infer(sent_vec,sent_len)
                seq_label=' '.join([id2label[str(e)] for e in seq[0]])
                print(seq_label)

    end_time=time.time()
    _logger.info("all_time:%s"%(end_time-start_time))
