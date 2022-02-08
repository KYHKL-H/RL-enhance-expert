# environment setting
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
import warnings
warnings.filterwarnings('ignore')

# external packages
import numpy as np
import json
import time
from tqdm import tqdm

# self writing files
import constants
from simulator import disease_model
from util import *

class Baselines(object):
    def __init__(self,
                MSA_name='Atlanta',
                vaccine_day=200,
                step_length=24,
                num_seed=100,
                numpy_seed=0,
                strategy='cases',
                vaccine_factor=1,
                infection_factor=1):
        super().__init__()

        print('Initializing...')
        # generate config
        config_data=locals()
        del config_data['self']
        del config_data['__class__']
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config_data['time']=time_data

        # environment
        self.numpy_seed=numpy_seed
        np.random.seed(self.numpy_seed)

        # loading data
        self.MSA_name=MSA_name
        self.data=load_data(self.MSA_name)
        self.poi_areas=self.data['poi_areas']
        self.cbg_ages=self.data['cbg_ages']# cbg*23groups
        self.cbg_sizes=self.data['cbg_sizes']
        self.poi_cbg_visits_list=self.data['poi_cbg_visits_list']# time_length*poi*cbg
        self.time_length=len(self.poi_cbg_visits_list)
        self.day_length=int(self.time_length/24)
        self.step_length=step_length

        assert len(self.cbg_ages)==len(self.cbg_sizes)
        assert len(self.cbg_ages)==self.poi_cbg_visits_list[0].shape[-1]
        self.num_cbg=len(self.cbg_ages)
        self.sum_population=np.sum(self.cbg_sizes)
        assert len(self.poi_areas)==self.poi_cbg_visits_list[0].shape[0]
        self.num_poi=len(self.poi_areas)

        # simulator
        self.num_seed=num_seed
        self.seeds=range(self.num_seed)
        self.infection_factor=infection_factor
        self.simulator=disease_model.Model(starting_seed=self.seeds,num_seeds=num_seed)
        self.simulator.init_exogenous_variables(poi_areas=self.poi_areas,
                                poi_dwell_time_correction_factors=self.data['poi_dwell_time_correction_factors'],
                                cbg_sizes=self.cbg_sizes,
                                poi_cbg_visits_list=self.poi_cbg_visits_list,
                                cbg_attack_rates_original = self.data['cbg_attack_rates_scaled']*self.infection_factor,
                                cbg_death_rates_original = self.data['cbg_death_rates_scaled'],
                                p_sick_at_t0=constants.parameters_dict[self.MSA_name][0],
                                home_beta=constants.parameters_dict[self.MSA_name][1],
                                poi_psi=constants.parameters_dict[self.MSA_name][2],
                                just_compute_r0=False,
                                latency_period=96,  # 4 days
                                infectious_period=84,  # 3.5 days
                                confirmation_rate=.1,
                                confirmation_lag=168,  # 7 days
                                death_lag=432)

        # dynamic features
        self.cbg_state_record=np.zeros((self.num_seed,self.step_length,3,self.num_cbg))

        # vaccine number
        self.vaccine_factor=vaccine_factor
        self.vacine_day=int((vaccine_day/37367)*self.sum_population)*self.vaccine_factor
        self.strategy=strategy

        print(f'Testing platform on {self.MSA_name} initialized')
        print(f'Number of CBGs={self.num_cbg}')
        print(f'Number of POIs={self.num_poi}')
        print(f'Total population={self.sum_population}')
        print(f'Time length={self.time_length}')
        print(f'Test with {self.num_seed} random seeds')
        print(f'Test with {self.strategy} strategy')

        # making output directory
        self.output_dir=os.path.join('..','model',f'{self.strategy}_results')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(os.path.join(self.output_dir,self.MSA_name+('' if self.vaccine_factor==1 else f'_{self.vaccine_factor}vaccine')+('' if self.infection_factor==1 else f'_{self.infection_factor}infection'))):
            os.mkdir(os.path.join(self.output_dir,self.MSA_name+('' if self.vaccine_factor==1 else f'_{self.vaccine_factor}vaccine')+('' if self.infection_factor==1 else f'_{self.infection_factor}infection')))
        self.output_dir=os.path.join(self.output_dir,self.MSA_name+('' if self.vaccine_factor==1 else f'_{self.vaccine_factor}vaccine')+('' if self.infection_factor==1 else f'_{self.infection_factor}infection'))
        with open(os.path.join(self.output_dir,'config.json'),'w') as f:
            json.dump(config_data,f)

        # results matrix
        self.result_C=np.empty((self.num_seed,self.day_length*self.step_length,self.num_cbg))
        self.result_D=np.empty((self.num_seed,self.day_length*self.step_length,self.num_cbg))
        self.result_vaccine=np.empty((self.num_seed,self.day_length-1,self.num_cbg),dtype=int)

    def vaccine_strategy(self,time_pointer):
        if self.strategy=='none':
            vaccine_mat=np.zeros((self.num_seed,self.num_cbg))

        elif self.strategy=='random':
            random_mat=np.random.rand(self.num_seed,self.num_cbg)
            random_mat=random_mat/np.sum(random_mat,axis=1,keepdims=True)
            vaccine_mat=self.vacine_day*random_mat

        elif self.strategy=='population':
            proportion_pop=self.cbg_sizes
            proportion_pop=proportion_pop/np.sum(proportion_pop)
            vaccine_mat=self.vacine_day*proportion_pop
            vaccine_mat=np.tile(vaccine_mat,(self.num_seed,1))

        elif self.strategy=='cases':
            proportion_pop=self.cbg_sizes
            proportion_pop=proportion_pop/np.sum(proportion_pop)
            proportion_cases=self.result_C[:,time_pointer*self.step_length-1,:]
            proportion_cases_sum=np.sum(proportion_cases,axis=1)
            proportion_cases=proportion_cases/proportion_cases_sum[:,np.newaxis]

            proportion_cases[proportion_cases_sum==0,:]=proportion_pop
            vaccine_mat=self.vacine_day*proportion_cases

        elif self.strategy=='new_cases':
            proportion_pop=self.cbg_sizes
            proportion_pop=proportion_pop/np.sum(proportion_pop)
            if time_pointer==1:
                proportion_cases=self.result_C[:,time_pointer*self.step_length-1,:]
            else:
                proportion_cases=self.result_C[:,time_pointer*self.step_length-1,:]-self.result_C[:,time_pointer*self.step_length-1-self.step_length,:]
            proportion_cases_sum=np.sum(proportion_cases,axis=1)
            proportion_cases=proportion_cases/proportion_cases_sum[:,np.newaxis]

            proportion_cases[proportion_cases_sum==0,:]=proportion_pop
            vaccine_mat=self.vacine_day*proportion_cases

        return vaccine_mat.astype(int)

    def update_cbg_state(self,current_C,current_D):
        self.cbg_state_record[:,:,1,:]=current_C
        self.cbg_state_record[:,:,2,:]=current_D
        self.cbg_state_record[:,:,0,:]=self.cbg_sizes-self.cbg_state_record[:,:,1,:]-self.cbg_state_record[:,:,2,:]

    def run(self):
        self.simulator.init_endogenous_variables()
        self.simulator.simulate_disease_spread(length=self.step_length,verbosity=1,no_print=True)
        current_C,current_D=self.simulator.output_record(full=False,length=self.step_length)
        self.update_cbg_state(current_C,current_D)

        self.result_C[:,:self.step_length,:]=current_C
        self.result_D[:,:self.step_length,:]=current_D

        self.simulator.empty_record()

        for day in tqdm(range(1,self.day_length)):
            vaccine_mat=self.vaccine_strategy(time_pointer=day)
            self.simulator.add_vaccine(vaccine_mat)
            self.simulator.simulate_disease_spread(length=24,verbosity=1,no_print=True)
            current_C,current_D=self.simulator.output_record(full=False,length=self.step_length)
            self.update_cbg_state(current_C,current_D)

            self.result_vaccine[:,day-1,:]=vaccine_mat
            self.result_C[:,day*self.step_length:day*self.step_length+self.step_length,:]=current_C
            self.result_D[:,day*self.step_length:day*self.step_length+self.step_length,:]=current_D

            self.simulator.empty_record()

        np.save(os.path.join(self.output_dir,'result_vaccine.npy'),self.result_vaccine)
        np.save(os.path.join(self.output_dir,'result_C.npy'),self.result_C)
        np.save(os.path.join(self.output_dir,'result_D.npy'),self.result_D)

if __name__ == '__main__':
    MSA_names=['Atlanta','Dallas','Miami']
    strategys=['random','population','cases','new_cases']
    for name in MSA_names:
        for strategy in strategys:
            baseline_platform=Baselines(MSA_name=name,strategy=strategy,infection_factor=3)
            baseline_platform.run()