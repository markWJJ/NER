'''
读取json 文件 或者将excel文件转化为标准的txt文件

'''

import json
import xlrd
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("data_deal")


class data_deal(object):

    def __init__(self):
        self.label_dict={'其他':0}
        self.label()


    def label(self):
        label_list=['身体部位','成分','抗炎抗菌','舒缓抗敏','清洁','美白亮肤','祛斑','抗氧化','去角质','控油','防晒','保湿柔润']

        for i in range(0,len(label_list)):
            self.label_dict[label_list[i]]=i+1



    def _deal_para(self,para_list):
        '''
        处理其中一个列表数据
        :param para_list:
        :return:
        '''
        result_data=[]

        try:
            result_data.append(para_list[0][1])
            result_data.append(para_list[2][1::])

            if len(para_list)<3:
                result_data.append(['None'+'_'+'None'])
            else:
                ss=[]
                ss.append([str(para_list[3][1])+'_'+str(para_list[3][2])])
                if len(para_list)>=5:
                    for e in para_list[4::]:
                        ss.append(str(e[1])+'_'+str(e[2]))
                    result_data.append(ss)
            return result_data
        except Exception as e:
            _logger.error('%s-%s'%(e,para_list[0][0]))


    def deal_excel(self,filename):
        '''
        处理excel文件
        :param filename:
        :return:
        '''
        excel_data=xlrd.open_workbook(filename)
        _logger.info("load data from %s"%filename)
        data=excel_data.sheet_by_name('Corpus-语料')
        wf=open('data.txt','w')
        data_all_list=[]
        data_list=[]

        for i in range(data.nrows):
            ele=data.row_values(i)
            if ele[1] != '' or ele[0] !='':
                ele=[e for e in ele if e!='']
                data_list.append(ele)
            else:
                if data_list != []:
                    data_all_list.append(data_list)
                    data_list=[]

        for ele in data_all_list:
            print(ele)
            try:
                wf.write(str(ele[0][0]))
                wf.write('\t\t')
                wf.write(str(ele[0][1]))
                wf.write('\t\t')
                wf.write(''.join([str(e) for e in ele[2][1::]]))
                wf.write('\t\t')


                if len(ele)<=3:
                    wf.write('None')
                    wf.write('\t\t')
                elif len(ele)==4:
                    wf.write(ele[3][1]+'_'+ele[3][2])
                    wf.write('\t\t')
                else:
                    wf.write(ele[3][1]+'_'+ele[3][2])
                    wf.write('\t\t')

                    for e in ele[4::]:
                        wf.write(str(e[0])+'_'+str(e[1]))
                        wf.write('\t\t')
            except Exception :
                _logger.error(ele)

            wf.write('\n')
        # for ele_ in data_all_list:
        #     ele=self._deal_para(ele_)
        #     print(ele)
        #     if ele and ele[2] is None:
        #         wf.write(ele[0])
        #         wf.write('\t\t')
        #         wf.write(''.join(ele[1]))
        #         wf.write('\t\t')
        #         wf.write(str(ele[3]))


    def deal_txt(self):
        '''
        对经 excel生成的txt进行再处理
        :return:
        '''
        wf=open('data_final.txt','w')
        with open('./data.txt','r') as rf:

            for line in rf.readlines():
                try:
                    lines=line.replace('\n','').split('\t\t')
                    lines=[e for e in lines if e!='']

                    sentence=lines[2]
                    wf.write(lines[0])
                    wf.write('\t\t')
                    wf.write(' '.join(list(sentence)))
                    wf.write('\t\t')
                    if lines[3]=='None':
                        wf.write(' '.join(['0']*len(sentence)))
                    else:
                        label_sent=[0]*len(sentence)
                        for labels in lines[3::]:
                            s=sentence.find(labels.split('_')[1])
                            lab=self.label_dict[labels.split('_')[0]]

                            for i in range(len(labels.split('_')[1])):
                                if i==0:
                                    label_sent[s+i]=str(lab)+'B'
                                elif i==len(labels.split('_')[1])-1:
                                    label_sent[s+i]=str(lab)+'E'
                                else:
                                    label_sent[s+i]=str(lab)+'M'


                        wf.write(' '.join([str(e) for e in label_sent]))
                    wf.write('\n')
                except Exception as e:
                    _logger.error('%s-%s'%(e,line))


    def deal_excel1(self,filename):
        '''
                处理excel文件
                :param filename:
                :return:
                '''
        excel_data = xlrd.open_workbook(filename)
        _logger.info("load data from %s" % filename)
        data = excel_data.sheet_by_name('Corpus-语料')
        wf = open('data1.txt', 'w')
        data_all_list = []
        data_list = []

        for i in range(data.nrows):
            ele = data.row_values(i)
            if ele[1] != '' or ele[0] != '':
                ele = [e for e in ele if e != '' and e!='label']
                data_list.append(ele)
            else:
                if data_list != []:
                    data_all_list.append(data_list)
                    data_list = []

        for ele in data_all_list:
            print(ele)
            try:
                words=ele[2][1::]
                labels=[0]*len(words)
                if len(ele)>=4:
                    for label_ in ele[3::]:
                        if label_:
                            label_id=self.label_dict[label_[0]]
                            start_id=int(label_[2])
                            for i in range(len(label_[1])):
                                if i==0:
                                    labels[i+start_id-1]=str(label_id)+'B'
                                elif i==len(label_[1])-1:
                                    labels[i+start_id-1]=str(label_id)+'E'
                                else:
                                    labels[i+start_id-1]=str(label_id)+'M'
                sentence=" ".join([str(e) for e in words])
                label=" ".join([str(e) for e in labels])
                wf.write(sentence)
                wf.write('\t\t')
                wf.write(label)
                wf.write('\n')
            except Exception as e:
                _logger.error('%s-%s'%(e,ele))



            # try:
            #     wf.write(str(ele[0][0]))
            #     wf.write('\t\t')
            #     wf.write(str(ele[0][1]))
            #     wf.write('\t\t')
            #     wf.write(''.join([str(e) for e in ele[2][1::]]))
            #     wf.write('\t\t')
            #
            #     if len(ele) <= 3:
            #         wf.write('None')
            #         wf.write('\t\t')
            #     elif len(ele) == 4:
            #         wf.write(ele[3][1] + '_' + ele[3][2])
            #         wf.write('\t\t')
            #     else:
            #         wf.write(ele[3][1] + '_' + ele[3][2])
            #         wf.write('\t\t')
            #
            #         for e in ele[4::]:
            #             wf.write(str(e[0]) + '_' + str(e[1]))
            #             wf.write('\t\t')
            # except Exception:
            #     _logger.error(ele)
            #
            # wf.write('\n')







if __name__ == '__main__':

    dd=data_deal()
    dd.deal_excel1('entity_1.xlsx')
    # dd.deal_txt()