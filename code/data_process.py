import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

from scipy import sparse
import pickle
import constants
from tqdm import tqdm

if __name__=='__main__':
    epic_data_root = '../data'
    MSA_list=constants.MSA_NAME_LIST

    for MSA_name in MSA_list:
        print(MSA_name)

        MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_name]
        f = open(os.path.join(epic_data_root, MSA_name, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
        poi_cbg_visits_list = pickle.load(f)
        f.close()

        for i in tqdm(range(len(poi_cbg_visits_list))):
            array=poi_cbg_visits_list[i].toarray()
            array[array<1e-3]=0
            array=sparse.csc_matrix(array)
            poi_cbg_visits_list[i]=array

        f = open(os.path.join(epic_data_root, MSA_name, '%s_2020-03-01_to_2020-05-02_processed.pkl'%MSA_NAME_FULL), 'wb') 
        pickle.dump(poi_cbg_visits_list,f)
        f.close()