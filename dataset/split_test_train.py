


import random
fw_dev=open('./dev_out_char.txt','w')
fw_train=open('./train_out_char.txt','w')

ss=[]
for line in open('./intent_data_char_all.txt','r').readlines():
    line=line.replace('\n','')
    lines=line.split('\t')
    label=lines[2].strip()
    ss.append(line)
ss=list(set(ss))
index=list(range(len(ss)))
random.shuffle(index)
res=[]
for id in index:
    res.append(ss[id])

split=0.2
dev_num=int(int(len(ss))*split)

for e in res[:dev_num]:
    fw_dev.write(e)
    fw_dev.write('\n')

for e in res[dev_num:]:
    fw_train.write(e)
    fw_train.write('\n')
