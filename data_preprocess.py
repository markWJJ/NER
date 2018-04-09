import numpy as np
import pickle
import os
global PATH
PATH=os.path.split(os.path.realpath(__file__))[0]
import logging
import util
import jieba
import re
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("r_net_data")

class Entity_Extration_Data(object):
    '''
    R_Net模型 数据处理模块
    '''
    def __init__(self,train_path,dev_path,test_path,batch_size,sent_len,flag):
        self.train_path=train_path  #训练文件路径
        self.dev_path=dev_path  #验证文件路径
        self.test_path=test_path    #测试文件路径
        self.sent_length=sent_len    # 文档长度
        self.batch_size=batch_size  #batch大小
        self.label_vocab={"0":0}
        ss=['B',"M","E"]
        index=1
        for i in range(1,13):
            for e in ss:
                key=str(i)+e
                self.label_vocab[key]=index
                index+=1
        if flag=="train_new":
            self.vocab=self.get_vocab()
            pickle.dump(self.vocab,open(PATH+"/vocab.p",'wb'))  # 词典
            pickle.dump(self.label_vocab,open(PATH+"/label_vocab.p",'wb'))  # 词典

        elif flag=="test" or flag=="train":
            self.vocab=pickle.load(open(PATH+"/vocab.p",'rb'))  # 词典
        self.index=0
        self.sent, self.sent_len, self.label,self.file_size,self.seg = self._train_data()
        if self.batch_size > self.file_size:
            _logger.error("batch规模大于训练数据规模！")

        self.id2word={}
        for k,v in self.vocab.items():
            self.id2word[v]=k

        self.id2label={}
        for k,v in self.label_vocab.items():
            self.id2label[str(v)]=k

    def get_vocab(self):
        '''
        构造字典 dict{NONE:0,word1:1,word2:2...wordn:n} NONE为未登录词
        :return: 
        '''
        train_file=open(self.train_path,'r')
        test_file=open(self.dev_path,'r')
        dev_file=open(self.test_path,'r')
        vocab={"NONE":0}
        label_vocab={}
        index=1
        label_index=0
        for ele in train_file:
            ele=ele.replace("\n","")
            eles=ele.split('\t\t')[0]
            labels=ele.split('\t\t')[1]
            ls=labels.split(' ')
            for label in ls:
                if label not in label_vocab:
                    label_vocab[label]=label_index
                    label_index+=1
            ws=eles.split(" ")
            for w in ws:
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1

        for ele in test_file:
            ele1=ele.replace("	"," ").replace("\n","")
            for w in ele1.split(" "):
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1

        for ele in dev_file:
            ele1=ele.replace("	"," ").replace("\n","")
            for w in ele1.split(" "):
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1

        train_file.close()
        dev_file.close()
        test_file.close()
        return vocab

    def seg_feature(self,seg_list):
        '''
        构建分词特征
        :param seg_list:
        :return:
        '''
        seg_fea=[]
        for e in seg_list:
            if len(e)==1:
                seg_fea.append(0)
            else:
                ss=[2]*len(e)
                ss[0]=1
                ss[-1]=3
                seg_fea.extend(ss)
        return seg_fea

    def _convert_sent(self,sent):
        '''
        将sent中的数字分开
        :param sent:
        :return:
        '''
        # pattern='\d.*'
        # seg = str(sent).replace("\n", "")
        # seg = ''.join([e for e in seg.split(' ')])
        # new_seg=[]
        # for e in seg:
        #     ee=re.sub(pattern,'NUM',e)
        #     new_seg.append(ee)
        sents = str(sent).replace("\n", "")
        sents = ''.join([e for e in sents.split(' ')])
        new_sent=[e for e in sents]
        return ' '.join(new_sent)


    def sent2vec(self,sent,max_len):
        '''
        根据vocab将句子转换为向量
        :param sent: 
        :return: 
        '''

        sent=self._convert_sent(sent)
        seg=str(sent).replace("\n","")


        seg=''.join([e for e in seg.split(' ')])
        seg_list=[e for e in jieba.cut(seg)]
        seg_list=self.seg_feature(seg_list)

        sent=str(sent).replace("\n","")
        sent_list=[]
        real_len=len(sent.split(" "))
        for word in sent.split(" "):
            word=word.lower()
            if word in self.vocab:
                sent_list.append(self.vocab[word])
            else:
                sent_list.append(0)

        if len(sent_list)>=max_len:
            new_sent_list=sent_list[0:max_len]
            new_seg_list=seg_list[0:max_len]
        else:
            new_sent_list=sent_list
            ss=[0]*(max_len-len(sent_list))
            new_sent_list.extend(ss)

            new_seg_list=seg_list
            ss_=[0]*(max_len-len(seg_list))
            new_seg_list.extend(ss_)

        sent_vec=np.array(new_sent_list)
        seg_vec=np.array(new_seg_list)

        if real_len>=max_len:
            real_len=max_len
        return sent_vec,real_len,seg_vec

    def get_ev_ans(self,sentence):
        '''
        获取 envience and answer_label
        :param sentence: 
        :return: 
        '''
        env_list=[]
        ans_list=[]
        for e in sentence.split(" "):
            try:
                env_list.append(e.split("/")[0])
                ans_list.append(self.label_dict[str(e.split("/")[1])])
            except:
                pass
        return " ".join(env_list),ans_list

    def _train_data(self):
        '''
        获取训练数据
        :return: 
        '''
        train_file = open(self.train_path, 'r')
        sent_list = []  # 句子的char list
        seg_list=[] #句子seg的list
        label_list = []  # label list
        file_size = 0
        sent_len_list = []
        for sentence in train_file.readlines():
            file_size+=1
            sentence = sentence.replace("\n", "")
            sentences = sentence.split("\t\t")

            sent_sentence = sentences[0]   # 分词的问句 sentence

            label_=[]
            for key in sentences[1].split(' '):

                if key in self.label_vocab:
                    label_.append(self.label_vocab[key])
                else:

                    label_.append(0)
            # print(label_)
            # print('\n')
            if len(label_)>=self.sent_length:
                label_=label_[:self.sent_length]
            else:
                ss=[0]*(self.sent_length-len(label_))
                label_.extend(ss)
            label = label_
            sent_vec, sent_real_len,seg_vec = self.sent2vec(sent_sentence, self.sent_length)

            sent_list.append(sent_vec)
            seg_list.append(seg_vec)
            sent_len_list.append(sent_real_len)
            label_list.append(label)
        train_file.close()
        result_sent = np.array(sent_list)
        result_sent_len_list = np.array(sent_len_list)
        result_label = np.array(label_list)
        return_seg=np.array(seg_list)
        sent,sent_len,label,seg = self.shuffle_(result_sent, result_sent_len_list, result_label,return_seg)[:]
        return sent,sent_len,label,file_size,seg

    def shuffle_(self,*args):
        '''
        将矩阵X打乱
        :param x: 
        :return: 
        '''
        ss=list(range(args[0].shape[0]))
        np.random.shuffle(ss)
        new_res=[]
        for e in args:
            new_res.append(np.zeros_like(e))
        fin_res=[]
        for index,ele in enumerate(new_res):
            for i in range(args[0].shape[0]):
                ele[i]=args[index][ss[i]]
            fin_res.append(ele)
        return fin_res

    def next_batch(self):
        '''
        获取的下一个batch
        :return: 
        '''
        num_iter=int(self.file_size/self.batch_size)
        if self.index<num_iter:
            return_sent=self.sent[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_sent_len=self.sent_len[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_label=self.label[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_seg=self.seg[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
        else:
            self.index=0
            return_sent=self.sent[0:self.batch_size]
            return_sent_len=self.sent_len[0:self.batch_size]
            return_label=self.label[0:self.batch_size]
            return_seg=self.seg[0:self.batch_size]

        return return_sent,return_sent_len,return_label,return_seg

    def get_dev(self,begin_id,end_id):
        '''
        读取验证数据集
        :return: 
        '''
        dev_file = open(self.dev_path, 'r')
        Q_list = []
        P_list = []
        Q_len_list=[]
        P_len_list=[]
        label_list = []
        train_sentcens = dev_file.readlines()
        for sentence in train_sentcens:
            sentence = sentence.replace("\n", "")
            sentences = sentence.split("\t\t")
            Q_sentence=sentences[0]
            P_sentence=sentences[1]
            label = [int(e) for e in sentences[2].split("-")]   # label
            Q_array,Q_real_len=self.sent2vec(Q_sentence,self.Q_length)
            P_array,P_real_len=self.sent2vec(P_sentence,self.P_length)
            Q_len_list.append(Q_real_len)
            P_len_list.append(P_real_len)
            Q_list.append(list(Q_array))
            P_list.append(list(P_array))
            label_list.append(label)
        dev_file.close()
        result_Q=np.array(Q_list)
        result_P=np.array(P_list)
        result_Q_len=np.array(Q_len_list)
        result_P_len=np.array(P_len_list)
        result_label=np.array(label_list)
        return result_Q[begin_id:end_id],result_P[begin_id:end_id],result_Q_len[begin_id:end_id],result_P_len[begin_id:end_id],result_label[begin_id:end_id]

    def get_test(self):
        '''
        读取测试数据集
        :return: 
        '''
        test_file = open(self.test_path, 'r')
        Q_list = []
        A_list = []
        label_list = []
        train_sentcens = test_file.readlines()
        for sentence in train_sentcens:
            sentences=sentence.split("	")
            Q_sentence=sentences[0]
            A_sentence=sentences[1]
            label=sentences[2]
            Q_array,_=self.sent2array(Q_sentence,self.Q_len)
            A_array,_=self.sent2array(A_sentence,self.P_len)

            Q_list.append(list(Q_array))
            A_list.append(list(A_array))
            label_list.append(int(label))
        test_file.close()
        result_Q=np.array(Q_list)
        result_A=np.array(A_list)
        result_label=np.array(label_list)
        return result_Q,result_A,result_label

    def get_sent(self,sentence):
        '''
        根据输入句子构建输入矩阵
        :param Q_sentence: 
        :return: 
        '''
        sentence=' '.join(list(sentence))
        sent_len=len(str(sentence).replace("\n","").split(" "))
        if sent_len>=self.sent_length:
            sent_len=self.sent_length
        sent_vec,_=self.sent2vec(sentence,self.sent_length)
        sent_len=np.array([sent_len])
        sent_vec=np.reshape(sent_vec,[1,sent_vec.shape[0]])
        return sent_vec,sent_len

    def get_infer_info(self,infer_dir):
        '''
        
        :param infer_dir: 
        :return: 
        '''
        infer_data=open(infer_dir,'r')
        Q_list = []  # 问句list
        P_list = []  # 文档list
        label_list = []  # label list
        file_size = 0
        Q_len_list = []
        P_len_list = []
        for sentence in infer_data.readlines():
            file_size += 1
            sentence = sentence.replace("\n", "")
            sentences = sentence.split("\t\t")

            Q_sentence = sentences[0]  # 分词的问句 sentence
            P_sentence = sentences[1]  # 分词的文档 sentence
            label = [int(e) for e in sentences[2].split("-")]  # label
            Q_vec, Q_real_len = self.sent2vec(Q_sentence, self.Q_length)
            P_vec, P_real_len = self.sent2vec(P_sentence, self.P_length)

            Q_list.append(Q_vec)
            P_list.append(P_vec)

            Q_len_list.append(Q_real_len)
            P_len_list.append(P_real_len)

            label_list.append(label)
        infer_data.close()
        result_Q = np.array(Q_list)
        result_P = np.array(P_list)
        result_Q_len_list = np.array(Q_len_list)
        result_P_len_list = np.array(P_len_list)

        result_label = np.array(label_list)

        Q, P, label, Q_len, P_len = self.shuffle_(result_Q, result_P, result_label, result_Q_len_list,
                                                  result_P_len_list)[:]
        return Q, P, label, Q_len, P_len, file_size


if __name__ == '__main__':

    dd = Entity_Extration_Data(train_path="./data/data1.txt", test_path="./data/test.txt",
                            dev_path="./data/dev.txt", batch_size=4 ,sent_len=30, flag="train_new")

    id2label=dd.id2label
    id2word=dd.id2word
    mm = util.Merge()
    _,_,_,seg=dd.next_batch()
    print(np.array(seg).shape)
    # for _ in range(10):
    #     sents,sent_lens,labels,seg=dd.next_batch()
    #
    #     for sent,label,sent_len in zip(sents,labels,sent_lens):
    #         sent=sent[:sent_len]
    #         label=label[:sent_len]
    #         sent=[id2word[e] for e in sent]
    #         label=[id2label[str(e)] for e in label]
    #
    #         rr_word=mm.merge_word(sent,label)
    #         print(rr_word)



    # print(dd.label_vocab)
    #
    # while True:
    #     sent=input('input:')
    #     sent_vec,sent_len=dd.get_sent(sent)
    #     print(sent_vec,sent_len.shape)