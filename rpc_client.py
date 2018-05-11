from xmlrpc.client import  ServerProxy
svr=ServerProxy("http://192.168.3.132:8083")
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
        res = svr.intent(step_data)
        print('all step:{}  this step:{}'.format(step,i))
        for k,v in res.items():
            fw.write(k)
            fw.write('\t')
            fw.write(' '.join([e[0] for e in v]))
            fw.write('\n')
        del res
        gc.collect()

if __name__ == '__main__':
    model_biaozhun('./信诚后台日志_处理.txt','./信诚后台日志_人工标注.txt')