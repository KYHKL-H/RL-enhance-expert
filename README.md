# Source code for paper "Reinforcement Learning Enhances the Experts: Large Scale COVID-19 Vaccines Allocation with Multi-factor Contact Network"

- These files are upload to Github and supplementary materials in the Microsoft CMT at the same time.
- Due to file size limitations, the folder '/data' only serves as a PLACEHOLDER and the data needed to be downloaded and put into it as the follow instruction.

# Implement steps
Here are the implement steps for Atlanta as an example.
1. Download SafeGraph Open Census Data from https://docs.safegraph.com/docs/open-census-data and replace the folder '/data/safegraph_open_census_data'. Actually, we only need one file in this dataset: '/data/safegraph_open_census_data/data/cbg_b01.csv'.
2. Download the IDs of the CBGs in Atlanta 'Atlanta_Sandy_Springs_Roswell_GA_cbg_ids.csv' from https://covid-mobility.stanford.edu//datasets/ [1] and put it into '/data/Atlanta/Atlanta_Sandy_Springs_Roswell_GA_cbg_ids.csv'.
3. Download the raw contact network data in Atlanta 'Atlanta_Sandy_Springs_Roswell_GA_2020-03-01_to_2020-05-02.pkl' from https://covid-mobility.stanford.edu//datasets/ [1] and put it into '/data/Atlanta/Atlanta_Sandy_Springs_Roswell_GA_2020-03-01_to_2020-05-02.pkl'.
4. 

# Reference
[1] Serina Y Chang*, Emma Pierson*, Pang Wei Koh*, Jaline Gerardin, Beth Redbird, David Grusky, and Jure Leskovec. "Mobility network models of COVID-19 explain inequities and inform reopening". Nature, 2020.
[2] Chen, Lin and Xu, Fengli and Han, Zhenyu and Tang, Kun and Hui, Pan and Evans, James and Li, Yong. "Strategic COVID-19 vaccine distribution can simultaneously elevate social utility and equity". arXiv preprint, 2021.
