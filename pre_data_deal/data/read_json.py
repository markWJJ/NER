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
for line in open('./all_baoxian_kg_entity_synonyms_alias.json','r'):
    ss=json.loads(line)
    ss=ss['_source']
    print(ss)
    stbz=ss['实体标准词']
    stlx=ss['实体类型']
    stty=ss['实体同义词']
    if stlx not in type_list:
        type_list.append(stlx)
    if stlx=='Jibing':
        Jibing_dict[stbz]=stty

    if stlx=='Qingjing':
        Qingjing_dict[stbz]=stty

    if stlx=='Baozhangxiangmu':
        Baozhangxiangmu_dict[stbz]=stty

    if stlx=='Shiyi':
        Shiyi_dict[stbz]=stty

    if stlx=='Didian':
        Didian_dict[stbz]=stty

    if stlx=='Yiyuan':
        Yiyuan_dict[stbz]=stty

    if stlx=='Yiyuandengji':
        Yiyuandengji_dict[stbz]=stty

    if stlx=='Baoxianzhonglei':
        Baoxianzhonglei_dict[stbz]=stty

    if stlx=='Jiaofeifangshi':
        Jiaofeifangshi_dict[stbz]=stty

    if stlx=='Baoxianchanpin':
        Baoxianchanpin_dict[stbz]=stty




print(type_list)



