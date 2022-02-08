# environment setting
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

import constants

epic_data_root = '../data'

MSA_names=['Atlanta','Dallas','Miami']
agg_result='community_index_fluid'

for MSA_name in MSA_names:
    print(MSA_name)
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_name]

    f = open(os.path.join(epic_data_root, MSA_name, '%s_2020-03-01_to_2020-05-02_processed.pkl'%MSA_NAME_FULL), 'rb') 
    poi_cbg_visits_list = pickle.load(f)
    f.close()

    num_poi=poi_cbg_visits_list[0].shape[0]
    num_cbg=poi_cbg_visits_list[0].shape[1]
    print(f'Number of CBGs={num_cbg}')

    c_index=np.load(os.path.join(epic_data_root, MSA_name, agg_result+'.npy'))
    num_community=np.max(c_index)+1
    print(f'Number of communities={num_community}')

    poi_cbg_visits_list_new=list()
    for i in tqdm(range(len(poi_cbg_visits_list))):
        poi_cbg_mat=poi_cbg_visits_list[i].toarray()
        poi_cbg_mat_new=np.empty((num_poi,num_community))
        for j in range(num_community):
            poi_cbg_mat_new[:,j]=np.sum(poi_cbg_mat[:,c_index==j],axis=1)
        poi_cbg_visits_list_new.append(sparse.csc_matrix(poi_cbg_mat_new))

    f = open(os.path.join(epic_data_root, MSA_name, '%s_2020-03-01_to_2020-05-02_processed_aggregate.pkl'%MSA_NAME_FULL), 'wb')
    pickle.dump(poi_cbg_visits_list_new,f)
    f.close()

    cbg_ids_msa = pd.read_csv(os.path.join(epic_data_root,MSA_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"CensusBlockGroup"}, inplace=True)
    cbg_ids_msa.insert(0,'community_index',c_index)

    # population in each community
    # Load SafeGraph data to obtain CBG sizes (i.e., populations)
    filepath = os.path.join(epic_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
    cbg_agesex = pd.read_csv(filepath)
    # Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
    cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, left_on='CensusBlockGroup',right_on='census_block_group', how='left')
    # Add up males and females of the same age, according to the detailed age list (DETAILED_AGE_LIST)
    # which is defined in constants.py
    for i in range(3,25+1): # 'B01001e3'~'B01001e25'
        male_column = 'B01001e'+str(i)
        female_column = 'B01001e'+str(i+24)
        cbg_age_msa[constants.DETAILED_AGE_LIST[i-3]] = cbg_age_msa.apply(lambda x : x[male_column]+x[female_column],axis=1)
    # Rename
    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
    # Extract columns of interest
    columns_of_interest = ['census_block_group','Sum'] + constants.DETAILED_AGE_LIST
    cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
    # Deal with NaN values
    cbg_age_msa.fillna(0,inplace=True)
    # Deal with CBGs with 0 populations
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)

    cbg_age_msa=pd.merge(cbg_age_msa,cbg_ids_msa,how='inner',left_on='census_block_group',right_on='CensusBlockGroup').groupby(by='community_index').sum()
    cbg_age_msa=cbg_age_msa.reset_index(level=['community_index'])
    cbg_age_msa.sort_values('community_index',ascending=True,inplace=True)
    community_sizes = cbg_age_msa['Sum'].values
    community_sizes = np.array(community_sizes,dtype='int32')
    np.save(os.path.join(epic_data_root, MSA_name, 'aggregate_sizes.npy'),community_sizes)

    community_ages = cbg_age_msa.to_numpy()[:,3:-1]/community_sizes[:,np.newaxis]
    np.save(os.path.join(epic_data_root, MSA_name, 'aggregate_ages.npy'),community_ages)