import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import torch
import pickle
import dgl
import pandas as pd
import numpy as np
from scipy import sparse

import constants

def array_norm(array,clip=100):
    data=array

    upper=np.percentile(data,clip)
    data_clip=np.clip(data,0,upper)

    mean=np.mean(data_clip)
    std=np.std(data_clip)
    data_norm=(data-mean)/std

    return data_norm

def sparse_mat_list_norm(sparse_mat_list,clip=100):
    data=list()
    for i in range(len(sparse_mat_list)):
        data.append(sparse_mat_list[i].data)
    data=np.concatenate(data)

    upper=np.percentile(data,clip)
    data=np.clip(data,0,upper)
    data_norm_factor=1/np.max(data)

    data_norm_list=list()
    for i in range(len(sparse_mat_list)):
        data_norm_list.append(sparse_mat_list[i]*data_norm_factor)

    return data_norm_list

def BuildGraph(poi_cbg):
    poi_num,cbg_num=poi_cbg.shape

    poi,cbg,weight=sparse.find(poi_cbg)
    poi=torch.tensor(poi)
    cbg=torch.tensor(cbg)
    weight=torch.tensor(weight,dtype=torch.float32)

    g=dgl.heterograph({('cbg','cbg_poi','poi'):(cbg,poi),
                        ('poi','poi_cbg','cbg'):(poi,cbg)})
    if (torch.max(cbg).item()!=cbg_num-1):
        g.add_nodes(cbg_num-1-torch.max(cbg).item(), ntype='cbg')
    if (torch.max(poi).item()!=poi_num-1):
        g.add_nodes(poi_num-1-torch.max(poi).item(), ntype='poi')

    g.edges['cbg_poi'].data['num']=weight
    g.edges['poi_cbg'].data['num']=weight

    g1=dgl.to_homogeneous(g, edata=['num'])
    edge_weight=g1.edata['num']
    g1.edata.pop('num')

    return g1,edge_weight

def load_data(MSA_name):
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_name]
    epic_data_root = '../data'
    data=dict()

    # Load POI-CBG visiting matrices
    f = open(os.path.join(epic_data_root, MSA_name, '%s_2020-03-01_to_2020-05-02_processed.pkl'%MSA_NAME_FULL), 'rb') 
    poi_cbg_visits_list = pickle.load(f)
    f.close()
    data['poi_cbg_visits_list']=poi_cbg_visits_list

    # Load precomputed parameters to adjust(clip) POI dwell times
    d = pd.read_csv(os.path.join(epic_data_root,MSA_name, 'parameters_%s.csv' % MSA_name)) 
    poi_areas = d['feet'].values
    poi_dwell_times = d['median'].values
    poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
    data['poi_areas']=poi_areas
    data['poi_times']=poi_dwell_times
    data['poi_dwell_time_correction_factors']=poi_dwell_time_correction_factors

    # Load CBG ids for the MSA
    cbg_ids_msa = pd.read_csv(os.path.join(epic_data_root,MSA_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)

    # Load SafeGraph data to obtain CBG sizes (i.e., populations)
    filepath = os.path.join(epic_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
    cbg_agesex = pd.read_csv(filepath)
    # Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
    cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
    del cbg_agesex
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

    # Obtain cbg sizes (populations)
    cbg_sizes = cbg_age_msa['Sum'].values
    cbg_sizes = np.array(cbg_sizes,dtype='int32')
    data['cbg_sizes']=cbg_sizes
    data['cbg_ages']=cbg_age_msa.to_numpy()[:,2:]/cbg_sizes[:,np.newaxis]

    ##############################################################################
    # Load and scale age-aware CBG-specific attack/death rates (original)

    cbg_death_rates_original = np.loadtxt(os.path.join(epic_data_root, MSA_name, 'cbg_death_rates_original_'+MSA_name))
    cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)

    # The scaling factors are set according to a grid search
    # Fix attack_scale
    attack_scale = 1
    cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
    cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_name]
    data['cbg_attack_rates_scaled']=cbg_attack_rates_scaled
    data['cbg_death_rates_scaled']=cbg_death_rates_scaled

    return data

def load_data_aggregate(MSA_name):
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_name]
    epic_data_root = '../data'
    data=dict()

    # Load POI-CBG visiting matrices
    f = open(os.path.join(epic_data_root, MSA_name, '%s_2020-03-01_to_2020-05-02_processed_aggregate.pkl'%MSA_NAME_FULL), 'rb') 
    poi_cbg_visits_list = pickle.load(f)
    f.close()
    data['poi_community_visits_list']=poi_cbg_visits_list

    data['community_sizes']=np.load(os.path.join(epic_data_root,MSA_name,'aggregate_sizes.npy'))
    data['community_ages']=np.load(os.path.join(epic_data_root,MSA_name,'aggregate_ages.npy'))
    data['community_index']=np.load(os.path.join(epic_data_root,MSA_name,'community_index_fluid.npy'))

    return data