import re

import os
PATH=os.path.split(os.path.realpath(__file__))[0]
import jieba
type_list=['Baozhangxiangmu', 'Jibing', 'Qingjing', 'Wenjian', 'Time', 'Jianejiaoqing', 'Jiaofeinianqi',
           'Shiyi', 'Baoxianzhonglei', 'Baoxianchanpin', 'Fenzhijigou', 'Didian', 'Yiyuan', 'Jiaofeifangshi',
           'Jine', 'Yiyuandengji', 'Baoxianjin', 'Jibingzhonglei', 'Hetonghuifu', 'Baodanjiekuan', 'Mianpeie']
for ele in type_list:
    jieba.load_userdict(PATH + '/data/%s.txt'%ele)
jieba.load_userdict(PATH+'/data/gsmz.txt')

jb=[e.replace('\n','') for e in open(PATH+'/data/Jibing.txt','r')] #疾病
bzxm=[e.replace('\n','') for e in open(PATH+'/data/Baozhangxiangmu.txt','r')] #保障项目
bxzl=[e.replace('\n','') for e in open(PATH+'/data/Baoxianzhonglei.txt','r')] #保险种类
st=[e.replace('\n','') for e in open(PATH+'/data/st.txt','r')] #身体部位
tjxm=[e.replace('\n','') for e in open(PATH+'/data/tjxm.txt','r')] #体检项目
pt=[e.replace('\n','') for e in open(PATH+'/data/pt.txt','r')] #平台
bxcp=[e.replace('\n','') for e in open(PATH+'/data/Baoxianchanpin.txt','r')] #保险产品
gs=[e.replace('\n','') for e in open(PATH+'/data/gs.txt','r')] #公司名
tsrq=[e.replace('\n','') for e in open(PATH+'/data/tsrq.txt','r')] #特殊人群
jbzl=[e.replace('\n','') for e in open(PATH+'/data/Jibingzhonglei.txt','r')] #疾病种类
qj=[e.replace('\n','') for e in open(PATH+'/data/Qingjing.txt','r')] #情景
jbhz=[e.replace('\n','') for e in open(PATH+'/data/jbhz.txt','r')] #疾病患者
fwxm=[e.replace('\n','') for e in open(PATH+'/data/fwxm.txt','r')] #服务项目
dd=[e.replace('\n','') for e in open(PATH+'/data/Didian.txt','r')] #地点
yy=[e.replace('\n','') for e in open(PATH+'/data/Yiyuan.txt','r')] #医院
jffs=[e.replace('\n','') for e in open(PATH+'/data/Jiaofeifangshi.txt','r')] #缴费方式
yydj=[e.replace('\n','') for e in open(PATH+'/data/Yiyuandengji.txt','r')] #医院等级
bxj=[e.replace('\n','') for e in open(PATH+'/data/Baoxianjin.txt','r')] #保险金
sy=[e.replace('\n','') for e in open(PATH+'/data/Shiyi.txt','r')] #释义
gsmc=[e.replace('\n','') for e in open(PATH+'/data/gsmz.txt','r')] #公司名称

entity_list=[]
entity_list.extend(jb)
entity_list.extend(bzxm)
entity_list.extend(bxzl)
entity_list.extend(bxcp)
entity_list.extend(jbzl)
entity_list.extend(sy)
entity_list.extend(tjxm)



class Intent_Data_Deal(object):

    def __init__(self):

        pass
    def deal_sent(self,line):
        '''
        对句子进行处理
        :param sent:
        :return:
        '''
        pattern = '\d{1,3}(\\.|，|、|？)'
        line = line.replace('\n', '')
        sent=line
        sent = re.subn(pattern, '', sent)[0]
        ss = []
        if sent in sy:
            ss.append('sy')
        else:
            sents=[e for e in jieba.cut(sent)]
            for e in sents:
                if e in jb:
                    ss.append('jb')
                elif e in bzxm:
                    ss.append('bzxm')
                elif e in bxzl:
                    ss.append('bxzl')
                elif e in bxcp:
                    ss.append('bxcp')
                elif e in qj:
                    ss.append('qj')
                elif e in bxj:
                    ss.append('bxj')
                elif e in fwxm:
                    ss.append('fwxm')
                else:
                    ss.append(e)
        sent=' '.join(ss)

        return sent


    def deal_sent_file(self,line):
        '''
        对句子进行处理
        :param sent:
        :return:
        '''
        pattern = '\d{1,3}(\\.|，|、|？)'
        line = line.replace('\n', '').strip()
        print([line])
        if '\t' in line:

            ll=str(line).split('\t')[1]
            sent=str(line).split('\t')[0]
        else:
            ll=str(line).split(' ')[1]
            sent=str(line).split(' ')[0]

        sent = re.subn(pattern, '', sent)[0]
        ss = []
        if sent in sy:
            ss.append('sy')
        else:
            sents=[e for e in jieba.cut(sent)]
            for e in sents:
                if e in jb:
                    ss.append('jb')
                elif e in bzxm:
                    ss.append('bzxm')
                elif e in bxzl:
                    ss.append('bxzl')
                elif e in bxcp:
                    ss.append('bxcp')
                elif e in qj:
                    ss.append('qj')
                elif e in bxj:
                    ss.append('bxj')
                elif e in fwxm:
                    ss.append('fwxm')
                else:
                    ss.append(e)
        sent=' '.join(ss)
        label = ll
        sent = 'BOS' + ' ' + sent + ' ' + 'EOS'
        entity = ' '.join(['O'] * len(sent.split(' ')))
        res = sent + '\t' + entity + '\t' + label

        sent = res.split('\t')[0]
        try:
            label = res.split('\t')[2]
        except:
            print(res)

        sent = sent.split(' ')
        ss = []
        for word in sent:
            word = word.lower()
            if word not in ['bzxm', 'jb', 'qj', 'bxj', 'bxcp', 'eos', 'bos', 'bxzl', 'sy', 'fwxm', 'bqx']:
                s = [e for e in word]
                ss.extend(s)
            else:
                ss.append(word)

        sents = ss
        slot = ['o'] * len(sents)

        sent = ' '.join(sents)
        slot = ' '.join(slot)

        return sent+'\t'+slot+'\t'+label

    def deal_file(self,input_file_name,out_file_name):
        '''
        将输入的标注数据 转换为带实体标签的char数据
        :param file_name:
        :return:
        '''
        fw=open(out_file_name,'w')

        for ele in open(input_file_name,'r').readlines():
            e=self.deal_sent_file(ele)
            fw.write(e)
            fw.write('\n')



if __name__ == '__main__':


    idd=Intent_Data_Deal()
    # print(idd.deal_sent('康爱保有什么保险责任吗'))
    idd.deal_file('整理标准.txt','../dataset/整理标准_out_char.txt')
