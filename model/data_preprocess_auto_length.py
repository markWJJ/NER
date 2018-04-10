import numpy as np
import pickle
import os
global PATH
PATH=os.path.split(os.path.realpath(__file__))[0]
import logging
import util
import jieba
import re
import random
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("r_net_data")

class Entity_Extration_Data(object):
    '''
    data deal use auto length
    '''
    def __init__(self,train_path,dev_path,test_path,batch_size,flag):
        self.train_path=train_path  #训练文件路径
        self.dev_path=dev_path  #验证文件路径
        self.test_path=test_path    #测试文件路径
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
        batch_list,self.num_batch=self.data_deal_train()
        self.batch_list=self.shuffle(batch_list)
        # # self.sent, self.sent_len, self.label,self.file_size,self.seg = self._train_data()
        # if self.batch_size > self.file_size:
        #     _logger.error("batch规模大于训练数据规模！")
        #
        self.id2word={}
        for k,v in self.vocab.items():
            self.id2word[v]=k

        self.id2label={}
        for k,v in self.label_vocab.items():
            self.id2label[str(v)]=k

    def shuffle(self,data_list):
        '''

        :param data_list:
        :return:
        '''
        index=[i for i in range(len(data_list))]
        random.shuffle(index)
        new_data_list=[data_list[e] for e in index]
        return new_data_list


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
        sents = str(sent).replace("\n", "")
        new_sent=[e for e in sents]
        return ' '.join(new_sent)

    def shuffle_sent(self,data_list):
        '''

        :param data_list:
        :return:
        '''
        index_list=[i for i in range(len(data_list))]
        random.shuffle(index_list)
        new_data_list=[data_list[i] for i in index_list]
        return new_data_list

    def padd_sentences(self,sent_list):
        '''
        find the max length from sent_list , and standardation
        :param sent_list:
        :return:
        '''
        words=[str(sent).replace('\n','').split('\t\t')[0] for sent in sent_list]
        labels=[str(sent).replace('\n','').split('\t\t')[1] for sent in sent_list]
        max_len=max([len(ele.split(' ')) for ele in words])

        word_arr=[]
        seg_arr=[]
        label_arr=[]
        real_len_arr=[]
        for sent,label in zip(words,labels):
            # sent = self._convert_sent(sent)
            # seg = str(sent).replace("\n", "")
            # seg_list = [e for e in jieba.cut(seg)]
            # seg_list = self.seg_feature(seg_list)

            sent_list = []
            real_len = len(sent.split(' '))
            for word in sent.split(' '):
                word = word.lower()
                if word in self.vocab:
                    sent_list.append(self.vocab[word])
                else:
                    sent_list.append(0)

            label_list=[]
            labell=label.split(' ')
            for ll in labell:
                if ll in self.label_vocab:
                    label_list.append(self.label_vocab[ll])
                else:
                    label_list.append(0)

            if len(sent_list) >= max_len:
                new_sent_list = sent_list[0:max_len]
                # new_seg_list = seg_list[0:max_len]
                new_label_list = label_list[0:max_len]
            else:
                new_sent_list = sent_list
                ss = [0] * (max_len - len(sent_list))
                new_sent_list.extend(ss)

                # new_seg_list = seg_list
                # ss_ = [0] * (max_len - len(seg_list))
                # new_seg_list.extend(ss_)

                new_label_list = label_list
                ss_l= [0]*(max_len-len(label_list))
                new_label_list.extend(ss_l)

            if real_len >= max_len:
                real_len = max_len

            real_len_arr.append(real_len)
            word_arr.append(new_sent_list)
            # seg_arr.append(new_seg_list)
            label_arr.append(new_label_list)


        real_len_arr=np.array(real_len_arr)
        word_arr=np.array(word_arr)
        # seg_arr=np.array(seg_arr)
        label_arr=np.array(label_arr)

        return word_arr,seg_arr,label_arr,real_len_arr


    def data_deal_train(self):
        '''

        :return:
        '''
        train_flie=open(self.train_path,'r')
        data_list=[line for line in train_flie.readlines()]

        data_list.sort(key=lambda x:len(x)) # sort not shuffle

        num_batch=int(len(data_list)/int(self.batch_size))

        batch_list=[]
        for i in range(num_batch):
            ele=data_list[i*self.batch_size:(i+1)*self.batch_size]
            word_arr, _, label_arr, real_len_arr=self.padd_sentences(ele)
            batch_list.append((word_arr,label_arr,real_len_arr))
        return batch_list,num_batch

    def next_batch(self):
        '''

        :return:
        '''
        num_iter = self.num_batch
        if self.index < num_iter:
            return_sent = self.batch_list[self.index][0]
            return_sent_len = self.batch_list[self.index][2]
            return_label = self.batch_list[self.index][1]
            current_length= self.batch_list[self.index][0].shape[1]
            current_length = np.array((current_length,),dtype=np.int32)
            self.index += 1
        else:
            self.index = 0
            return_sent = self.batch_list[self.index][0]
            return_sent_len = self.batch_list[self.index][2]
            return_label = self.batch_list[self.index][1]
            current_length = np.array((len(self.batch_list[self.index][0]),),dtype=np.int32)


        return return_sent, return_sent_len, return_label,current_length





if __name__ == '__main__':

    dd = Entity_Extration_Data(train_path="./data1.txt", test_path="./test.txt",
                            dev_path="./dev.txt", batch_size=4 , flag="train_new")

    for _ in range(10):
        sent,sent_len,label,lene=dd.next_batch()
        print(lene)
