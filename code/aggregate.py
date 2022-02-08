# environment setting
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from networkx.algorithms.community import asyn_fluidc
from scipy import sparse

import constants

if len(sys.argv)==1:
    MSA_name = 'Atlanta'
else:
    MSA_name = sys.argv[1]

MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_name]
epic_data_root = '../data'

f = open(os.path.join(epic_data_root, MSA_name, '%s_2020-03-01_to_2020-05-02_processed.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()
cbg_num=poi_cbg_visits_list[0].shape[1]

G=nx.Graph()
G_mat=np.zeros((cbg_num,cbg_num))

for poi_cbg_visit in tqdm(poi_cbg_visits_list):
    poi,cbg,weight=sparse.find(poi_cbg_visit)
    poi_sort_index=np.argsort(poi)
    poi=poi[poi_sort_index]
    cbg=cbg[poi_sort_index]
    weight=weight[poi_sort_index]

    encounter_start=poi[0]
    encounter_start_index=0
    for i in range(len(poi)):
        encounter_end=poi[i]
        encounter_end_index=i
        if encounter_end==encounter_start:
            pass
        else:
            encounter_cbgs=cbg[encounter_start_index:encounter_end_index]
            encounter_weights=weight[encounter_start_index:encounter_end_index]
            for i1 in range(len(encounter_cbgs)):
                for i2 in range(i1+1,len(encounter_cbgs)):
                    encounter_product=encounter_weights[i1]*encounter_weights[i2]
                    if encounter_product>=10:
                        G_mat[encounter_cbgs[i1],encounter_cbgs[i2]]+=encounter_product
            encounter_start=poi[i]
            encounter_start_index=i

    encounter_cbgs=cbg[encounter_start_index:]
    encounter_weights=weight[encounter_start_index:]
    for i1 in range(len(encounter_cbgs)):
        for i2 in range(i1+1,len(encounter_cbgs)):
            encounter_product=encounter_weights[i1]*encounter_weights[i2]
            if encounter_product>=10:
                G_mat[encounter_cbgs[i1],encounter_cbgs[i2]]+=encounter_product

for i1 in range(cbg_num):
    G.add_node(i1)
    for i2 in range(i1+1,cbg_num):
        if G_mat[i1][i2]>0:
            G.add_edge(i1,i2,weight=G_mat[i1][i2])

num_community=int(np.ceil(cbg_num/500))
c=asyn_fluidc(G,num_community)
c_index=np.zeros(cbg_num,dtype=int)
c=list(c)
for i in range(len(c)):
    community=c[i]
    for cbg in community:
        c_index[cbg]=i
np.save(os.path.join(epic_data_root, MSA_name, 'community_index_fluid.npy'),c_index)
print(f'Fluid | community={len(c)}')
print(f'Number of CBGs={cbg_num}')