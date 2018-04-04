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
    sent_len=30    # 问句长度
    embedding_dim=50    #词向量维度
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
    epoch=200
    lambda1=0.01

config=Config()
tf.app.flags.DEFINE_float("lambda1", config.lambda1, "l2学习率")
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_float("keep_dropout", config.keep_dropout, "dropout")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("sent_len", config.sent_len, "句子长度")
tf.app.flags.DEFINE_integer("embedding_dim", config.embedding_dim, "词嵌入维度.")
tf.app.flags.DEFINE_integer("hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("epoch", config.epoch, "每轮训练迭代次数")
tf.app.flags.DEFINE_string("summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_string("mod", "infer", "默认为训练") # true for prediction
FLAGS = tf.app.flags.FLAGS




class Bilstm(object):

    def __init__(self,num_class,hidden_dim,seq_len,seq_dim,mode,batch_size,embedding_dim,vocab_num,crf_mode,loss_weight_mode):
        self.num_class=num_class
        self.hidden_dim=hidden_dim
        self.seq_len=seq_len
        self.seq_dim=seq_dim
        self.mode=mode
        self.batch_size=batch_size
        self.seq_vec=tf.placeholder(shape=(None,),dtype=tf.int32)
        self.loss_weight_mode=loss_weight_mode
        self.embedding_dim=embedding_dim
        #if self.loss_weight_mode:
        # self.loss_weight=tf.placeholder(shape=(None,),dtype=tf.int32)
        self.embedding=tf.Variable(tf.random_uniform(shape=(vocab_num,embedding_dim),minval=-1.0,maxval=1.0,dtype=tf.float32))
        self.X_sent=tf.placeholder(shape=(None,self.seq_len),dtype=tf.int32)

        self.input_emb=tf.nn.embedding_lookup(self.embedding,self.X_sent)

        self.Y = tf.placeholder(shape=(None,self.seq_len), dtype=tf.int32)

        X_=tf.transpose(self.input_emb,[1,0,2])
        X_= tf.unstack(X_,self.seq_len,0)
        self.crf_mode = crf_mode

        self.lstm_input=X_
        #self.lstm_input=tf.unstack(self.lstm_input,seq_len,1)

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

            if self.mode == "lstm":
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

            elif self.mode == "lstm_attention":
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

            elif self.mode == "lstm_crf":
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

            elif self.mode=="lstm_crf_new":
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

    def crf_target_score(self,observation_maxtrix,probability_matrix_hot):
        '''
        根据observation 计算target_score
        :return: 
        '''
        # probability_matrix[None,seq_len+1,num_class+1]
        # observation_maxtrix[batch_szie,seq_len+1,1]
        #transfer_matrix_ = tf.unstack(self.transfer_matrix, self.num_class + 1, 0)

        observation_maxtrix_hot=tf.one_hot(observation_maxtrix,self.num_class+1,1,0,2)#[batch_size,seq_len+1,num_class+1]
        observation_maxtrix_hot_flatten=tf.cast(tf.reshape(observation_maxtrix_hot,[-1]),tf.float32)
        probability_matrix_hot_flatten=tf.cast(tf.reshape(probability_matrix_hot,[-1]),tf.float32)
        point_score=tf.reduce_sum((probability_matrix_hot_flatten*observation_maxtrix_hot_flatten))




        probability_matrix_ = tf.transpose(probability_matrix_hot, [1, 2, 0])
        probability_matrix_ = tf.unstack(probability_matrix_, self.seq_len + 1, 0)
        # path = [tf.constant(self.num_class+1, shape=(tf.shape(probability_matrix_), 1), dtype=tf.int32)]

        observation_maxtrix_=tf.transpose(observation_maxtrix,[1,0,2])
        observation_maxtrix_=tf.unstack(observation_maxtrix_,self.seq_len+1,0)

        #all_pro=tf.cast(tf.reshape(observation_maxtrix_[0],(1,-1)),dtype=tf.float32)
        
        
        tran_score=[]
        for i in range(1,len(observation_maxtrix_)):
            # probability_matrix_[i] shape(num_class+1,batch_size)
            # observation_maxtrix_[i] shape=[batch_size,1]
            tran_pro= tf.gather(self.transfer_matrix,observation_maxtrix_[i-1]) #shape=(batch_size,num_class+1)
            tran_pro=tf.reshape(tran_pro,(self.num_class+1,-1)) #shape=(num_class+1,batch_size)
            tran_pro=tf.gather(tran_pro,observation_maxtrix_[i]) #(1,batch_size)

            tran_score.append(tran_pro)
        #res_pro=(tf.reduce_mean(all_pro))/(self.seq_len+1)
        all_pro=tf.stack(tran_score,0)
        target_score=point_score+tf.reduce_sum(all_pro)



        return target_score

    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self,probability_matrix):
        '''
        根据前馈算法以及crf的概率矩阵和转移矩阵计算路径 probability_matrix[None,seq_len+1,num_class+1]
        :param probability_matrix: 
        :param transfer_matrix: 
        :return: 
        '''
        probability_matrix=tf.reshape(probability_matrix,[-1,self.seq_len+1,self.num_class+1,1])
        probability_matrix=tf.transpose(probability_matrix,[1,0,2,3])
        previous = probability_matrix[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1,self.seq_len+1):
            previous=tf.reshape(previous,[-1,self.num_class+1,1])
            current=tf.reshape(probability_matrix[t,:,:,:],[-1,1,self.num_class+1])
            alpha_t=previous+current+self.transfer_matrix
            alpha_t=tf.cast(alpha_t,dtype=tf.float32)
            max_scores.append(tf.reduce_max(alpha_t, reduction_indices=2))
            max_scores_pre.append(tf.argmax(alpha_t, dimension=2))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=2), [-1, self.num_class+1, 1])
            #alpha_t = tf.reshape(tf.reduce_max(alpha_t, reduction_indices=2), [-1, self.num_class+1, 1])
            previous = alpha_t
            alphas.append(alpha_t)

        #alphas = tf.reshape(tf.concat(0, alphas), [self.seq_len+1, -1,self.num_class+1, 1])
        last_alphas=alphas[-1]
        last_alphas = tf.reshape(last_alphas, [-1, self.num_class+1, 1])

        #max_scores = tf.reshape(tf.concat(0, max_scores), (self.seq_len + 1, -1, self.num_class+1))
        max_scores =tf.stack(max_scores,0)
        max_scores_pre = tf.stack(max_scores_pre,0)
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        #return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre

        return tf.reduce_sum(last_alphas), max_scores, max_scores_pre

    def crf(self,softmax_lstm,crf_mode):
        '''
        输入经lstm得到的n个类别的概率，进行crf处理,softmax.shape=[None,seq_len,num_class]
        :param lstm_outs: 
        :return: 
        '''
        # 构建概率矩阵
        probability_matrix=softmax_lstm
        # 构建转移矩阵
        self.transfer_matrix=tf.Variable(tf.random_uniform(shape=(self.num_class+1,self.num_class+1),minval=0.0,maxval=1.0))
        # 增加一个标注，在矩阵前增加一个padd，构建新的概率矩阵probability_matrix
        sl=tf.argmax(softmax_lstm,2)
        padd=tf.zeros_like(sl,dtype=tf.float32)
        padd=tf.reshape(padd,(-1,self.seq_len,1))
        #padd=tf.constant(0.0,shape=(self.batch_size,self.seq_len,1),dtype=tf.float32)
        probability_matrix_orgin=tf.concat((probability_matrix,padd),2) #shape=[None,seq_len,num+1]
        #len_padd=tf.constant(self.num_class,shape=(self.batch_size,1,1))
        sll=tf.argmax(sl,1)
        len_padd=tf.ones_like(sll,dtype=tf.int32)*self.num_class
        len_padd=tf.reshape(len_padd,(-1,1,1))
        len_padd_hot=tf.one_hot(indices=len_padd,depth=self.num_class+1,axis=2,on_value=1,off_value=0)
        len_padd_hot=tf.cast(len_padd_hot,dtype=tf.float32)
        len_padd_hot=tf.reshape(len_padd_hot,shape=(-1,1,self.num_class+1))
        # 在句首增加一个词
        probability_matrix=tf.concat((len_padd_hot,probability_matrix_orgin),1)#shape=[None,seq_len+1,num+1]
        total_path_score,max_score,max_socre_pre=self.forward(probability_matrix)

        if crf_mode=="train":
            # 观测矩阵
            observation_maxtrix_orgin=tf.reshape(self.Y,shape=[-1,self.seq_len,1])
            observation_maxtrix=tf.concat((len_padd,observation_maxtrix_orgin),1)#[None,seq_len+1,1]
            self.observation_maxtrix1=observation_maxtrix
            # 生成概率
            observation_maxtrix_hot = tf.one_hot(indices=observation_maxtrix, depth=self.num_class+1, on_value=1, off_value=0, axis=2)#[None,seq_len,numclass+1]
            observation_maxtrix_hot = tf.cast(observation_maxtrix_hot, dtype=tf.int32)

            #loss=tf.losses.sparse_softmax_cross_entropy(labels=observation_maxtrix_hot,logits=best_path_hot)

            #print(loss)
            target_score=self.crf_target_score(observation_maxtrix=observation_maxtrix,probability_matrix_hot=probability_matrix)
            loss=-(target_score-total_path_score)
            #loss=tf.abs(target_score-total_path_score)

            return loss,max_score,max_socre_pre

        elif crf_mode=="decode":
            return max_score,max_socre_pre

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

    def comput_acc(self,lstm_predict,label):
        '''
        计算正确率
        :param lstm_predict: 
        :param label: 
        :return: 
        '''
        lstm_predict_max=np.argmax(lstm_predict,2)
        lstm_predict_flatten=list(lstm_predict_max.flatten())
        label_flatten=list(np.array(label).flatten())
        ss=sum([1 for i,j in zip(lstm_predict_flatten,label_flatten) if int(i)==int(j) ])
        length=sum([1 for i in lstm_predict_flatten if int(i)!=0])
        acc=(float(ss)/float(len(label_flatten)))
        #acc=(float(ss)/float(length))
        return acc

    def train_1(self,dd):
            config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                    inter_op_parallelism_threads=16,
                                    intra_op_parallelism_threads=16,
                                    log_device_placement=False)
            saver=tf.train.Saver()
            self.crf_mode="train"
            with tf.Session(config=config) as sess:
                # sess.run(tf.global_variables_initializer())
                saver.restore(sess,'./save_model/lstm_model.ckpt')
                # seq_vec = np.array([ModelConfig.max_length] * ModelConfig.batch_size, dtype=np.int32)
                # dev_sent,dev_entity,dev_label,dev_loss_weight=dd.get_dev()
                # dev_seq_vec = np.array([ModelConfig.max_length] * dev_sent.shape[0], dtype=np.int32)
                init_train_loss=999.99
                init_dev_loss=999.99
                init_train_acc=0.0
                init_dev_acc=0.0
                for i in range(5000):

                    sent,sent_len, label=dd.next_batch()

                    # train_loss,_,train_predict=sess.run([self.loss_op,self.optimizer,self.lstm_prediction],feed_dict={
                    #                                                         self.X_sent:sent,
                    #                                                           self.Y:label,
                    #                                                            self.seq_vec:sent_len,
                    # })

                    train_loss, _,  = sess.run([self.loss_op, self.optimizer],
                                                            feed_dict={
                                                                self.X_sent: sent,
                                                                self.Y: label,
                                                                self.seq_vec: sent_len,
                                                            })
                    print("这是第%s次迭代"%i)
                    print(train_loss)
                    # train_acc = self.comput_acc(train_predict, label)
                    # print(train_loss,train_acc)
                    # train_predict_label=np.argmax(train_predict,2)
                    # for e1,e2 in zip(train_predict_label,label):
                    #     print(' '.join([str(e) for e in e1]))
                    #     print(' '.join(str(e)for e in e2))
                    #     print(train_predict_label)
                    #     print(label)
                        # print('\n\n\n')
                    # dev_loss, dev_predict = sess.run([self.loss_op, self.lstm_prediction],
                    #                                  feed_dict={self.X_sent: dev_sent,
                    #                                             self.Y:dev_label,
                    #                                             self.seq_vec: dev_seq_vec,
                    #                                             self.loss_weight:dev_loss_weight})
                    # dev_acc = self.comput_acc(dev_predict, dev_label)
                    #if dev_acc>init_dev_acc and train_acc>init_train_acc:
                    if train_loss < init_train_loss:

                        init_train_loss=train_loss
                        saver.save(sess,'./save_model/lstm_model.ckpt')
                        _logger.info('save')
                        # init_dev_acc=dev_acc
                        # print("训练误差%s,准确率%s   验证误差%s,准确率%s"%(train_loss,train_acc,dev_loss,dev_acc))
                        # print("save model")
                        # if self.mode=="lstm":
                        #     saver.save(sess,"./save_model/lstm_model.ckpt")
                        # elif self.mode=="lstm_attention":
                        #     saver.save(sess,"./save_model/lstm_attention_model.ckpt")
                        # elif self.mode=="lstm_crf":
                        #     saver.save(sess,"./save_model/lstm_crf_model.ckpt")

    def viterbi(self, max_scores, max_scores_pre,seq_len,predict_size=None):
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][seq_len-1])
            # last_max_node = 0
            for t in range(seq_len)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]
            best_paths.append(path)
        return best_paths

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

    def crf_train(self,dd):
            config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                    inter_op_parallelism_threads=16,
                                    intra_op_parallelism_threads=16,
                                    log_device_placement=False)
            saver=tf.train.Saver()
            self.crf_mode="train"
            wf=open('./out.txt','w')
            with tf.Session(config=config) as sess:
                saver.restore(sess,"./save_model/lstm_crf.ckpt")
                # sess.run(tf.global_variables_initializer())
                # saver.restore(sess, "./save_model/lstm_crf_new1.model")

                # dev_sent,dev_entity,dev_label,dev_loss_weight,dev_seq_vec=dd.get_dev()
                init_train_loss=999.99
                init_dev_loss=999.99
                init_train_acc=0.0
                init_dev_acc=0.0
                for i in range(5000):
                    sent,sent_len, label=dd.next_batch()

                    '''
                    train_loss,_,transfer_matrix,max_score,max_socre_pre=sess.run([self.loss_op,self.optimizer,self.transfer_matrix,
                                                                                   self.max_score,self.max_score_pre],
                                                                                  feed_dict={self.X_sent:sent,
                                                                              self.Y:label,
                                                                               self.seq_vec:train_real_len,
                                                                                self.loss_weight:loss_weight})
                    '''

                    train_loss,_,logit,trans_params=sess.run([self.loss_op,self.optimizer,self.logit,self.trans_params],feed_dict={
                                                                            self.X_sent:sent,
                                                                              self.Y:label,
                                                                               self.seq_vec:sent_len,
                                                                                  })

                    # dev_loss,dev_logit,dev_trans_params=sess.run([self.loss_op,self.logit,self.trans_params],feed_dict={self.X_sent:dev_sent,
                    #                                                           self.Y:dev_label,
                    #                                                            self.seq_vec:dev_seq_vec,
                    #                                                                self.loss_weight:dev_loss_weight})


                    verbit_seq=self.Verbit(logits=logit,batch_size=sent.shape[0],trans_params=trans_params,sequence_lengths=sent_len)
                    # dev_verbit_seq=self.Verbit(logits=dev_logit,batch_size=dev_sent.shape[0],trans_params=dev_trans_params,sequence_lengths=dev_seq_vec)

                    for e,e1 in zip(label,verbit_seq):
                        e=" ".join([str(e_) for e_ in e])
                        e1=' '.join([str(e1_) for e1_ in e1])
                        wf.write(e)
                        wf.write('\n')
                        wf.write(e1)
                        wf.write('\n')
                        wf.write('\n')



                    crf_acc_train=self.crf_acc(verbit_seq,label,sent_len)

                    print("This is %s iter"%i)
                    print("训练误差,训练准确率",train_loss,crf_acc_train)
                    # print("验证误差,验证准确率",dev_loss,crf_acc_dev)
                    if train_loss<init_train_loss :
                        print("save model")
                        init_train_loss=train_loss
                        saver.save(sess,"./save_model/lstm_crf.ckpt")

                    print("\n")



                    '''
                    best_path=self.viterbi(max_score,max_socre_pre,60,ModelConfig.batch_size)
                    crf_acc_train=self.crf_acc(best_path,label)
                    print(best_path)
                    #origin_path=np.array(origin_path)
                    #origin_path=np.reshape(origin_path,(16,61))
                    #print(origin_path)
                    print("--------------------")
                    print("这是第%s次迭代"%i)
                    #train_acc = self.comput_acc(train_predict, label)
                    dev_batch_size=dev_sent.shape[0]
                    dev_loss,dev_max_score,dev_max_score_pre= sess.run([self.loss_op, self.max_score,self.max_score_pre],
                                                     feed_dict={self.X_sent: dev_sent,
                                                                self.Y:dev_label,
                                                                self.seq_vec: dev_seq_vec,
                                                                self.loss_weight:dev_loss_weight})
                    dev_best_path=self.viterbi(dev_max_score,dev_max_score_pre,60,dev_batch_size)

                    crf_acc_dev=self.crf_acc(dev_best_path,dev_label)
                    print("训练误差%s,准确率%s   验证误差%s,准确率%s" % (train_loss, crf_acc_train, dev_loss, crf_acc_dev))
                    if crf_acc_dev>init_dev_acc and crf_acc_train>init_train_acc:
                        init_train_acc=crf_acc_train
                        init_dev_acc=crf_acc_dev
                        print("save model")
                        if self.mode=="lstm":
                            saver.save(sess,"./save_model/lstm_model.ckpt")
                        elif self.mode=="lstm_attention":
                            saver.save(sess,"./save_model/lstm_attention_model.ckpt")
                        elif self.mode=="lstm_crf":
                            saver.save(sess,"./save_model/lstm_crf_model.ckpt")
                            #pass
                    '''

    def crf_test(self,sent_array,seq_len):
        '''
        crf解码预测
        :param sent_array: 
        :return: 
        '''

        config = tf.ConfigProto(device_count={"CPU": 16},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=16,
                                intra_op_parallelism_threads=16,
                                log_device_placement=False)
        saver = tf.train.Saver()
        test_batch_size=sent_array.shape[0]
        seq_vec = np.array([ModelConfig.max_length] * test_batch_size, dtype=np.int32)
        #self.crf_mode = "decode"
        with tf.Session(config=config) as sess:
            print("load model!")
            saver.restore(sess, "./save_model/lstm_crf_new1.model")

            #max_score, max_score_pre= sess.run([self.max_score, self.max_score_pre], feed_dict={self.X_sent: sent_array,
            #                                              self.seq_vec: seq_vec})
            #best_path = self.viterbi(max_score, max_score_pre, 60, test_batch_size)
            logit, trans_params = sess.run([self.logit, self.trans_params],
                                                          feed_dict={self.X_sent: sent_array,
                                                                     self.seq_vec:seq_len ,
                                                                     })
            verbit_seq = self.Verbit(logits=logit, batch_size=sent_array.shape[0], trans_params=trans_params,
                                     sequence_lengths=seq_len)


            return verbit_seq

    def lstm_atten_train(self,dd):
            config = tf.ConfigProto(device_count={"CPU": 16},  # limit to num_cpu_core CPU usage
                                    inter_op_parallelism_threads=16,
                                    intra_op_parallelism_threads=16,
                                    log_device_placement=False)
            saver=tf.train.Saver()
            self.crf_mode="train"
            with tf.Session(config=config) as sess:
                #saver.restore(sess,"./save_model/lstm_attention_model.ckpt")
                sess.run(tf.global_variables_initializer())
                seq_vec = np.array([ModelConfig.max_length] * ModelConfig.batch_size, dtype=np.int32)

                dev_sent,dev_entity,dev_label,dev_loss_weight=dd.get_dev()
                dev_seq_vec = np.array([ModelConfig.max_length] * dev_sent.shape[0], dtype=np.int32)
                init_train_loss=999.99
                init_dev_loss=999.99
                init_train_acc=0.0
                init_dev_acc=0.0
                for i in range(5000):

                    sent, entity, label,loss_weight=dd.next_batch()

                    train_loss,_,train_predict=sess.run([self.loss_op,self.optimizer,self.lstm_prediction],feed_dict={self.X_sent:sent,
                                                                              self.Y:label,
                                                                               self.seq_vec:seq_vec,
                                                                                self.loss_weight:loss_weight})
                    print("这是第%s次迭代"%i)
                    train_acc = self.comput_acc(train_predict, label)
                    dev_loss, dev_predict = sess.run([self.loss_op, self.lstm_prediction],
                                                     feed_dict={self.X_sent: dev_sent,
                                                                self.Y:dev_label,
                                                                self.seq_vec: dev_seq_vec,
                                                                self.loss_weight:dev_loss_weight})
                    dev_acc = self.comput_acc(dev_predict, dev_label)
                    print("训练误差%s,准确率%s   验证误差%s,准确率%s" % (train_loss, train_acc, dev_loss, dev_acc))
                    if dev_acc>=init_dev_acc and train_acc>=init_train_acc:
                    #if train_acc > init_train_acc:

                        init_train_acc=train_acc
                        init_dev_acc=dev_acc
                        print("save model")
                        if self.mode=="lstm":
                            saver.save(sess,"./save_model/lstm_model.ckpt")
                        elif self.mode=="lstm_attention":
                            saver.save(sess,"./save_model/lstm_attention_model.ckpt")
                        elif self.mode=="lstm_crf":
                            saver.save(sess,"./save_model/lstm_crf_model.ckpt")

    def test_1(self,sent_array,entity_array):

        config = tf.ConfigProto(device_count={"CPU": 16},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=16,
                                intra_op_parallelism_threads=16,
                                log_device_placement=False)
        saver = tf.train.Saver()
        leng=sent_array.shape[0]
        seq_vec = np.array([ModelConfig.max_length] * leng, dtype=np.int32)
        self.crf_mode="decode"
        with tf.Session(config=config) as sess:
            print("load model!")
            if self.mode=="lstm":
                saver.restore(sess,"./save_model/lstm_model.ckpt")
            elif self.mode=="lstm_attention":
                saver.restore(sess,"./save_model/lstm_attention_model.ckpt")
            elif self.mode == "lstm_crf":
                saver.save(sess, "./save_model/lstm_crf_model.ckpt")

            lstm_predect=sess.run([self.lstm_prediction],feed_dict={self.X_sent:sent_array,
                                                                self.seq_vec:seq_vec})

            xx = lstm_predect[0]
            ss = []
            for i in range(xx.shape[0]):
                ss.append(np.argmax(xx[i], 1))
            return ss 



if __name__ == '__main__':
    start_time=time.time()
    with tf.device("/cpu:0"):

        _logger.info("load data")
        dd = data_preprocess.Entity_Extration_Data(train_path="./data/data1.txt", test_path="./data/test.txt",
                            dev_path="./data/dev.txt", batch_size=FLAGS.batch_size ,sent_len=FLAGS.sent_len, flag="train_new")
        num_class=len(dd.label_vocab)
        vocab_num=len(dd.vocab)
        nn_model = Bilstm(hidden_dim=FLAGS.hidden_dim,
                      seq_len=FLAGS.sent_len, seq_dim=FLAGS.embedding_dim,
                      num_class=num_class,
                      mode="lstm_crf_new",batch_size=FLAGS.batch_size,crf_mode='train',loss_weight_mode='train',vocab_num=vocab_num
                          ,embedding_dim=FLAGS.embedding_dim)

        _logger.info("load data finish")
        nn_model.crf_train(dd)

    end_time=time.time()
    _logger.info("all_time:%s"%(end_time-start_time))