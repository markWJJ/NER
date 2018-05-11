import json



Jibing_dict={} #疾病
Qingjing_dict={} #情景
Baozhangxiangmu_dict={} #保障项目
Shiyi_dict={} #释义
Didian_dict={} #地点
Yiyuan_dict={} #医院
Yiyuandengji_dict={} #医院等级
Baoxianzhonglei_dict={} #保险种类
Jiaofeifangshi_dict={} #缴费方式
Baoxianchanpin_dict={} #保险产品

type_list=[]
file_r=open('./all_baoxian_kg_entity_synonyms_alias.json','r')

for line in file_r:
    ss=json.loads(line)
    ss=ss['_source']

    stbz=ss['实体标准词']
    stlx=ss['实体类型']
    stty=ss['实体同义词']
    if stlx not in type_list:
        type_list.append(stlx)
file_r.close()
print(type_list)

for ele in type_list:
    fw=open('%s.txt'%ele,'w')
    for line in open('./all_baoxian_kg_entity_synonyms_alias.json','r'):
        ss = json.loads(line)
        ss = ss['_source']

        stbz = ss['实体标准词']
        stlx = ss['实体类型']
        stty = ss['实体同义词']
        if ele==stlx:
            fw.write(stbz)
            fw.write('\n')
            for e in stty:
                fw.write(e)
                fw.write('\n')
