from xmlrpc.client import  ServerProxy
import requests
import sys
sys.path.append('..')
from sklearn.metrics import classification_report,precision_recall_fscore_support

svr_cnn=ServerProxy("http://192.168.3.132:8083")
svr_lstm=ServerProxy('http://192.168.3.132:8084')
# svr_re=ServerProxy('http://192.168.3.132:8086')

import json
import gc
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_biaozhu")



def model_biaozhun(origin_file,biaozhun_file):
    '''
    模
    :param origin_file:
    :param biaozhun_file:
    :return:
    '''
    print('开始自动标注')
    fw=open('./%s'%biaozhun_file,'w')
    fr=open('./%s'%origin_file,'r').readlines()
    print('load origin file')
    file_length=len(fr)
    fr_=fr[:]
    del fr
    gc.collect()
    max_len=3000
    step=int(file_length/max_len)+1
    for i in range(step):

        step_data=[line.replace('\n','') for line in fr_[i*max_len:(i+1)*max_len]]
        res = svr_lstm.intent(step_data)
        print('all step:{}  this step:{}'.format(step,i))
        for k,v in res.items():
            fw.write(k)
            fw.write('\t')
            fw.write(' '.join([e[0] for e in v]))
            fw.write('\n')
        del res
        gc.collect()



def toupiao(file):

    ss_sent=[]
    ss_label=[]
    for ele in open(file,'r').readlines():
        sent=ele.split('\t')[0]
        label=ele.split('\t')[2].replace('\n','').strip()
        sent=''.join([e for e in sent.split(' ')])
        ss_sent.append(sent)
        ss_label.append(label.split(' ')[0])
    print(len(ss_sent))
    # res_cnn = svr_cnn.intent(ss_sent)
    res_lstm=svr_lstm.intent(ss_sent)

    res=[e[0][0] for e in res_lstm]

    ss=classification_report(ss_label,res)
    print(ss)

    # for k,v in res_cnn.items():
    #     label_cnn=v
    #     label_lstm=res_lstm[k]
    #     label_re=res_re[k]
    #     cnn_list=[e[0] for e in label_cnn]
    #     lstm_list=[e[0] for e in label_lstm]
    #     re_list=[e[0] for e in label_re]
    #
    #     ss_dict={}
    #     for e in cnn_list:
    #         if e not in ss_dict:
    #             ss_dict[e]=1
    #         else:
    #             ss=ss_dict[e]
    #             ss+=1
    #             ss_dict[e]=ss
    #
    #
    #
    #
    #     print(k)
    #     print('cnn'," ".join([e[0] for e in label_cnn]))
    #
    #     print('lstm'," ".join([e[0] for e in label_lstm]))
    #
    #
    #     print('re'," ".join([e[0] for e in label_re]))
    #
    #     print('\n')


    # res = requests.post('http://192.168.3.132:8086/intent', json={'data': ss_sent})
    # res.json()
    # for e in res:
    #     print(type(e))
    #     s=json.loads(e)
    #     print(s)
    #     for key in s:
    #         print(res_cnn[key])
    #         print(res_lstm[key])
    #         print(s[key])

    # for k,v in res_cnn.items():
    #     label_cnn=[e[0] for e in v]
    #     label_lstm=[e[0] for e in res_lstm[k]]
    #
    #     print(k)
    #     print(label_cnn)
    #     print(label_lstm)
    #     print('\n')



if __name__ == '__main__':
    # model_biaozhun('./信诚后台日志_处理.txt','./信诚后台日志_人工标注.txt')
    #
    dev_file='./dataset/dev_out_char.txt'
    toupiao(dev_file)
