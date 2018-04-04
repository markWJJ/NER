import numpy as np



class Eval(object):

    def __init__(self,act_data,pre_data):
        self.act_data=np.array(act_data)
        self.pre_data=np.array(pre_data)

        act_dict_N={}
        act_dict_R={}

        pre_dict_N={}
        pre_dict_R={}

        for act_e,pre_e in zip(self.act_data,self.pre_data):
            for act_ele,pre_ele in zip(act_e,pre_e):

                if act_ele != pre_ele:
                    if str(act_ele) not in act_dict_N:

                        act_dict_N[str(act_ele)]=1
                    else:
                        num = act_dict_N[str(act_ele)]
                        num += 1
                        act_dict_N[str(act_ele)] = num

                    if str(pre_ele) not in pre_dict_N:
                        pre_dict_N[str(pre_ele)] = 1
                    else:
                        num = pre_dict_N[str(pre_ele)]
                        num += 1
                        pre_dict_N[str(pre_ele)] = num

                else:
                    if str(act_ele) not in act_dict_R:
                        act_dict_R[str(act_ele) ] = 1
                    else:
                        num = act_dict_R[str(act_ele)]
                        num += 1
                        act_dict_R[str(act_ele) ] = num

                    if str(pre_ele) not in pre_dict_R:
                        pre_dict_R[str(pre_ele)] = 1
                    else:
                        num = pre_dict_R[str(pre_ele)]
                        num += 1
                        pre_dict_R[str(pre_ele)] = num



        self.act_dict_N=act_dict_N
        self.act_dict_R=act_dict_R

        self.pre_dict_N=pre_dict_N
        self.pre_dict_R=pre_dict_R

    def precision(self,k):
        '''
        计算精度
        :return:
        '''
        k=str(k)

        if k in self.pre_dict_R:
            v=self.pre_dict_R[k]
        else:
            v=0

        if k not in self.pre_dict_N and v!=0:
            precision=1.0
        elif k not in self.pre_dict_N and v==0:
            precision=0.0
        else:
            precision = float(v) / float(v + self.pre_dict_N[k])
        return precision


    def recall(self,k):
        k=str(k)
        if k in self.act_dict_R:
            v=self.act_dict_R[k]
        else:
            v=0

        if k not in self.act_dict_N and v !=0:
            recall=1.0
        elif k not in self.act_dict_N and v ==0:
            recall=0.0
        else:
            recall = float(v) / float(v + self.act_dict_N[k])
        return recall

    def f1_score(self,k):
        k=str(k)
        precision=self.precision(k)
        recall=self.recall(k)
        f1=(2*precision*recall)/float(precision+recall)
        return f1


class Merge(object):

    def __init__(self):


        self.id2name={'1':'身体部位',
                      '2':'成分',
                      '3':'抗炎抗菌',
                      '4':'舒缓抗敏',
                      '5':'清洁',
                      '6':'美白亮肤',
                      '7':'祛斑',
                      '8':'抗氧化',
                      '9':'去角质',
                      '10':'控油',
                      '11':'防晒',
                      '12':'保湿柔润'
                      }


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
                if labels[i] == '0':
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
                    sss+="_"+self.id2name[labels[i][:-1]]
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
                    sss+="_"+self.id2name[labels[i][:-1]]
                    new_words.append(sss)
                else:
                    s=words[i]+'_'+self.id2name[labels[i][:-1]]
                    new_words.append(s)
                    f += 1
        return " ".join(new_words)




if __name__ == '__main__':

    act=[[0,1,2,3,4,0,1,2,4,0,0],
         [1,2,4,1,0,0,0,1,3,3,2]]

    pre=[[0,0,1,3,4,1,1,3,4,0,0],
         [1,2,3,3,0,0,2,0,3,3,2]]

    ev=Eval(act,pre)
    print(ev.precision(1))
    print(ev.recall(1))
    print(ev.f1_score(1))

