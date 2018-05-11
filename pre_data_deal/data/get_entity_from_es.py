
import requests
r = requests.post('http://192.168.3.105:9200/all_baoxian_kg_entity_synonyms_alias/_search', json={
    'size' : 10000,
    'query': {
        'match_all' : {}
    }
})
data = r.json()
all_datas = [t['_source'] for t in data['hits']['hits']]
#
# bzxm_list=[]
# jb_list=[]
# qj_list=[]
# sy_list=[]
# bxcp_list=[]
# fwxm_list=[]
# yy_list=[]
# bxzl_list=[]
# jffs_list=[]
# bxj_list=[]
# jbzl_list=[]
# # 'Jibing', 'Qingjing', 'Wenjian', 'Shiyi', 'Jine', 'Baoxianchanpin', 'Mianpeie', 'Huiyuandengji', 'Fuwuxiangmu', 'Fenzhijigou', 'Didian', 'Yiyuan', 'Hetonghuifu', 'Yiyuandengji', 'Baoxianzhonglei', 'Baodanjiekuan', 'Jiaofeifangshi', 'Baoxianjin', 'Jibingzhonglei'
# ss_dict_list=[]
# for ele in all_datas:
#     print(ele)
#     if ele['实体类型']=='Jibingzhonglei':
#         bzxm_list.append(ele['实体标准词'])
#         for ee in ele['实体同义词']:
#             if ee:
#                 bzxm_list.append(ee)
#
# bzxm_list=list(set(bzxm_list))
#
# for ele in bzxm_list:
#     print(ele)
#




ss=[]
for line in open('./Shiyi.txt','r').readlines():
    ss.append(line)

ss=list(set(ss))

for line in ss:
    print(line.replace('\n',''))


