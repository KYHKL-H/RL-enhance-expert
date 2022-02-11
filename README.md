# Source code for paper "Reinforcement Learning Enhances the Experts: Large Scale COVID-19 Vaccines Allocation with Multi-factor Contact Network"

- These files are upload to Github and supplementary materials in the Microsoft CMT at the same time.
- Due to file size limitations, the folder '/data' only serves as a PLACEHOLDER and the data needed to be downloaded and put into it as the follow instruction.

# Implement steps
Here are the implement steps for Atlanta as an example.
1. Download SafeGraph Open Census Data from https://docs.safegraph.com/docs/open-census-data and replace the folder '/data/safegraph_open_census_data'. Actually, we only need one file in this dataset: '/data/safegraph_open_census_data/data/cbg_b01.csv'.
2. Download the IDs of the CBGs in Atlanta 'Atlanta_Sandy_Springs_Roswell_GA_cbg_ids.csv' from https://covid-mobility.stanford.edu//datasets/ [1] and place it as '/data/Atlanta/Atlanta_Sandy_Springs_Roswell_GA_cbg_ids.csv'.
3. Download the raw contact network data in Atlanta 'Atlanta_Sandy_Springs_Roswell_GA_2020-03-01_to_2020-05-02.pkl' from https://covid-mobility.stanford.edu//datasets/ [1] and place it as '/data/Atlanta/Atlanta_Sandy_Springs_Roswell_GA_2020-03-01_to_2020-05-02.pkl'.
4. Get the average dwelling time and floor space of POIs in Atlanta '/data/Atlanta/parameters_Atlanta.csv' with the method in [1].
5. Get the CBG specific death rate in Atlanta '/data/Atlanta/cbg_death_rates_original_Atlanta' with method in [2].
6. Run '/code/data_process.py' and get the processed contact network data in Atlanta '/data/Atlanta/Atlanta_Sandy_Springs_Roswell_GA_2020-03-01_to_2020-05-02_processed.pkl'.
7. Run '/code/aggregate.py' and get the community division indexes in Atlanta '/data/Atlanta/community_index_fluid.npy'.
8. Run '/code/process_aggregate.py' and get the aggregated contact network data in Atlanta '/data/Atlanta/Atlanta_Sandy_Springs_Roswell_GA_2020-03-01_to_2020-05-02_processed_aggregate.pkl' and the corresponding sizes and age structures of the communities '/data/Atlanta/aggregate_sizes.npy', '/data/Atlanta/aggregate_ages.npy'
9. Run '/code/train.py' for model training and the model are saved in '/model/Atlanta' for testing and analyzing.
10. Run '/code/baseline.py' for baseline results roll out.

Python files in '/code' not mentioned above are imported by the mentioned ones.

# Reference
[1] Serina Y Chang*, Emma Pierson*, Pang Wei Koh*, Jaline Gerardin, Beth Redbird, David Grusky, and Jure Leskovec. "Mobility network models of COVID-19 explain inequities and inform reopening". Nature, 2020.

[2] Lin Chen, Fengli Xu, Zhenyu Han, Kun Tang, Pan Hui, James Evans, and Yong Li. "Strategic COVID-19 vaccine distribution can simultaneously elevate social utility and equity". arXiv preprint, 2021.
