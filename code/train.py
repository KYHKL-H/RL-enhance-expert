# environment setting
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
import warnings
warnings.filterwarnings('ignore')

# external packages
import numpy as np
import torch
import json
import copy
import time
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.nn import DataParallel
from tqdm import tqdm

# self writing files
import constants
import networks
from reward_fun import reward_fun
from simulator import disease_model
from util import *

class GraphRL(object):
    def __init__(self,
                MSA_name='Atlanta',
                vaccine_day=200,
                step_length=24,
                num_seed=128,
                num_episode=300,
                num_epoch=10,
                batch_size=512,
                buffer_size=8192,
                save_interval=1,
                lr=1e-4,
                soft_replace_rate=0.5,
                gamma=0.6,
                epsilon=0.2,
                SPR_weight=2,
                manual_seed=0):
        super().__init__()

        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        del config_data['__class__']
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config_data['time']=time_data

        # environment
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.manual_seed=manual_seed
        torch.manual_seed(self.manual_seed)
        if self.device=='cuda':
            torch.cuda.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        dgl.seed(self.manual_seed)

        # loading cbg data (for simulation)
        self.MSA_name=MSA_name
        self.data=load_data(self.MSA_name)
        self.poi_areas=self.data['poi_areas']
        self.poi_times=self.data['poi_times']
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
        self.batch_size=batch_size
        self.seeds=range(self.num_seed)
        self.simulator=disease_model.Model(starting_seed=self.seeds,num_seeds=self.num_seed)
        self.simulator.init_exogenous_variables(poi_areas=self.poi_areas,
                                poi_dwell_time_correction_factors=self.data['poi_dwell_time_correction_factors'],
                                cbg_sizes=self.cbg_sizes,
                                poi_cbg_visits_list=self.poi_cbg_visits_list,
                                cbg_attack_rates_original = self.data['cbg_attack_rates_scaled'],
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

        # loading community data
        self.data_community=load_data_aggregate(self.MSA_name)
        self.community_ages=self.data_community['community_ages']# cbg*23groups
        self.community_sizes=self.data_community['community_sizes']
        self.community_index=self.data_community['community_index']
        self.poi_community_visits_list=self.data_community['poi_community_visits_list']# time_length*poi*cbg

        assert len(self.community_ages)==len(self.community_sizes)
        assert len(self.community_ages)==self.poi_community_visits_list[0].shape[-1]
        assert len(self.community_index)==self.num_cbg
        self.num_community=len(self.community_ages)

        # dynamic features
        self.community_state_num=np.zeros((self.num_seed,3,self.num_community))#S,C,D
        self.community_state_record=np.zeros((self.num_seed,self.step_length,3,self.num_community))
        self.community_casediff=np.zeros((self.num_seed,3,self.num_community))

        self.cbg_case_num=np.zeros((self.num_seed,self.num_cbg))
        self.cbg_case_num_old=np.zeros((self.num_seed,self.num_cbg))

        # static features (normalized)
        self.community_ages_norm=array_norm(self.community_ages,clip=100)# cbg*23
        self.poi_areas_norm=array_norm(self.poi_areas,clip=99)[:,np.newaxis]# poi*1
        self.poi_times_norm=array_norm(self.poi_times,clip=99)[:,np.newaxis]# poi*1
        self.community_ages_norm=torch.FloatTensor(self.community_ages_norm).to(self.device)
        self.poi_areas_norm=torch.FloatTensor(self.poi_areas_norm).to(self.device)
        self.poi_times_norm=torch.FloatTensor(self.poi_times_norm).to(self.device)

        self.community_ages_norm_repeated_batch=self.community_ages_norm.repeat(self.batch_size,1,1)
        self.poi_areas_norm_repeated_batch=self.poi_areas_norm.repeat(self.batch_size,1,1)
        self.poi_times_norm_repeated_batch=self.poi_times_norm.repeat(self.batch_size,1,1)
        self.community_ages_norm_repeated_seed=self.community_ages_norm.repeat(self.num_seed,1,1)
        self.poi_areas_norm_repeated_seed=self.poi_areas_norm.repeat(self.num_seed,1,1)
        self.poi_times_norm_repeated_seed=self.poi_times_norm.repeat(self.num_seed,1,1)

        # network features (normalized)
        poi_community_visits_day=list()
        for i in range(self.day_length):
            poi_community_visits_day.append(np.sum(self.poi_community_visits_list[i*24:i*24+24]))

        self.poi_visit_day=list()
        for i in range(len(poi_community_visits_day)):
            self.poi_visit_day.append(np.sum(poi_community_visits_day[i],axis=1))
        self.poi_visit_day=np.array(self.poi_visit_day)
        self.poi_visit_day_norm=array_norm(self.poi_visit_day,clip=100)
        self.poi_visit_day_norm=torch.FloatTensor(self.poi_visit_day_norm).to(self.device)

        poi_community_visits_day_norm=sparse_mat_list_norm(poi_community_visits_day,clip=99)
        self.poi_community_visits_day_network=list()
        print('Building graphs...')
        for i in tqdm(range(len(poi_community_visits_day_norm))):
            net,edge_weights=BuildGraph(poi_community_visits_day_norm[i])
            self.poi_community_visits_day_network.append((net.to(self.device),edge_weights.to(self.device)))

        # vaccine number
        self.vacine_day=int((vaccine_day/37367)*self.sum_population)

        # replay buffers
        self.buffer_size=buffer_size
        self.buffer_pointer=0
        self.buffer_sa=np.zeros((self.buffer_size,self.step_length,3,self.num_community))
        self.buffer_sadiff=np.zeros((self.buffer_size,3,self.num_community))
        self.buffer_sb=np.zeros((self.buffer_size),dtype=int)
        self.buffer_a=np.zeros((self.buffer_size,self.num_community))
        self.buffer_r=np.zeros(self.buffer_size)
        self.buffer_sa1=np.zeros((self.buffer_size,self.step_length,3,self.num_community))
        self.buffer_sadiff1=np.zeros((self.buffer_size,3,self.num_community))
        self.buffer_sb1=np.zeros((self.buffer_size),dtype=int)
        self.buffer_logp=np.zeros(self.buffer_size)

        # training trackors
        self.num_episode=num_episode
        self.num_epoch=num_epoch
        self.save_interval=save_interval
        self.episode_deaths_trackor=list()
        self.episode_cases_trackor=list()
        self.critic_loss_trackor=list()
        self.actor_loss_trackor=list()
        self.SPR_loss_trackor=list()

        # networks
        self.online_encoder=DataParallel(networks.Encoder(step_length=self.step_length).to(self.device))
        self.online_GCN=networks.GCN().to(self.device)
        self.online_projector=DataParallel(networks.Projector(num_cbg=self.num_community).to(self.device))
        self.transition=DataParallel(networks.Transition().to(self.device))
        self.predictor=DataParallel(networks.Predictor().to(self.device))
        self.PPO_actor=DataParallel(networks.Actor(num_cbg=self.num_community).to(self.device))
        self.PPO_critic=DataParallel(networks.Critic(num_cbg=self.num_community).to(self.device))

        self.target_encoder=copy.deepcopy(self.online_encoder)
        self.target_encoder.eval()
        self.target_GCN=copy.deepcopy(self.online_GCN)
        self.target_GCN.eval()
        self.target_projector=copy.deepcopy(self.online_projector)
        self.target_projector.eval()

        self.soft_replace_rate=soft_replace_rate
        self.gamma=gamma
        self.epsilon=epsilon
        self.SPR_weight=SPR_weight

        # optimizers
        self.lr=lr
        self.online_encoder_opt=torch.optim.Adam(self.online_encoder.parameters(),lr=self.lr)
        self.online_GCN_opt=torch.optim.Adam(self.online_GCN.parameters(),lr=self.lr)
        self.transition_opt=torch.optim.Adam(self.transition.parameters(),lr=self.lr)
        self.online_projector_opt=torch.optim.Adam(self.online_projector.parameters(),lr=self.lr)
        self.predictor_opt=torch.optim.Adam(self.predictor.parameters(),lr=self.lr)
        self.PPO_actor_opt=torch.optim.Adam(self.PPO_actor.parameters(),lr=self.lr)
        self.PPO_critic_opt=torch.optim.Adam(self.PPO_critic.parameters(),lr=self.lr)

        print(f'Training platform on {self.MSA_name} initialized')
        print(f'Number of communities={self.num_community}')
        print(f'Number of POIs={self.num_poi}')
        print(f'Total population={self.sum_population}')
        print(f'Time length={self.time_length}')
        print(f'Train with {self.num_seed} random seeds')

        # making output directory
        self.output_dir=os.path.join('..','model',f'{self.MSA_name}_aggregate_{self.num_seed}seeds_{time_data}')
        os.mkdir(self.output_dir)
        with open(os.path.join(self.output_dir,'config.json'),'w') as f:
            json.dump(config_data,f)

    def test_simulation(self):
        for num in range(1):
            self.simulator.reset_random_seed()
            self.simulator.init_endogenous_variables()
            # mat=500*np.ones((self.num_seed,self.num_cbg))
            # mat[:30,:]-=300
            # self.simulator.add_vaccine(mat)
            for i in range(63):
                # if i==20:
                #     mat=500*np.ones((self.num_seed,self.num_cbg))
                #     self.simulator.add_vaccine(mat)
                self.simulator.simulate_disease_spread(no_print=True)

            T1,L_1,I_1,R_1,C2,D2,total_affected, history_C2, history_D2, total_affected_each_cbg=self.simulator.output_record(full=True)

            gt_result_root=os.path.join('..','model','simulator_test')
            if not os.path.exists(gt_result_root):
                os.mkdir(gt_result_root)
            savepath = os.path.join(gt_result_root, f'cases_cbg_no_vaccination_{self.MSA_name}_{self.num_seed}seeds_step_raw{num}a.npy')
            np.save(savepath, history_C2)
            savepath = os.path.join(gt_result_root, f'deaths_cbg_no_vaccination_{self.MSA_name}_{self.num_seed}seeds_step_raw{num}a.npy')
            np.save(savepath, history_D2)

    def test_network(self):
        community_state_record=torch.FloatTensor(self.community_state_record).to(self.device)
        print(community_state_record.shape)
        community_statediff=community_state_record[:,-1,:,:].unsqueeze(1)
        print(community_statediff.shape)

        index=[0]*self.num_seed
        g,edge_weight,poi_visits=self.get_indexed_vectors(index)
        cbg_encode,poi_encode=self.online_encoder(community_state_record,community_statediff,self.community_ages_norm_repeated_seed,poi_visits,self.poi_areas_norm_repeated_seed,self.poi_times_norm_repeated_seed)
        print(cbg_encode.shape)
        print(poi_encode.shape)
        cbg_embeddings=self.online_GCN(g,edge_weight,cbg_encode,poi_encode)
        print(cbg_embeddings.shape)

        mu,sigma=self.PPO_actor(cbg_embeddings)
        value=self.PPO_critic(cbg_embeddings)
        print(mu.shape)
        print(sigma.shape)
        print(value.shape)

        action,_=self.get_action(sigma,mu)
        print(action.shape)

        cbg_embeddings_new=self.transition(cbg_embeddings,action)
        print(cbg_embeddings_new.shape)
        projection1=self.online_projector(cbg_embeddings_new)
        projection2=self.online_projector(cbg_embeddings)
        print(projection1.shape)
        print(projection2.shape)
        prediction=self.predictor(projection1)
        print(prediction.shape)

    def get_indexed_vectors(self,index):
        g_list=list()
        edge_weight_list=list()
        for i in range(len(index)):
            g,edge_weight=self.poi_community_visits_day_network[index[i]]
            g_list.append(g)
            edge_weight_list.append(edge_weight)

        poi_visits=self.poi_visit_day_norm[index,:]

        return g_list,edge_weight_list,poi_visits

    def update_community_state(self,current_C,current_D):
        self.community_state_record[:,:,1,:]=current_C
        self.community_state_record[:,:,2,:]=current_D
        self.community_state_record[:,:,0,:]=self.community_sizes-self.community_state_record[:,:,1,:]-self.community_state_record[:,:,2,:]

        self.community_state_num=self.community_state_record[:,-1,:,:]

    def get_action(self,mu,sigma,action=None):
        batch_size=len(mu)
        eye=torch.eye(self.num_community).to(self.device).repeat(batch_size,1,1)
        sigma_mat=eye*(torch.square(sigma).unsqueeze(1))
        dist = MultivariateNormal(mu, sigma_mat)

        if action==None:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action,log_prob

        else:
            log_prob = dist.log_prob(action)
            return log_prob

    def get_vaccine(self,actions):
        # actions=actions-2*(np.min(actions,axis=1)[:,np.newaxis])
        actions=np.exp(actions)
        actions=actions/(np.sum(actions,axis=1)[:,np.newaxis])
        vaccine_mat_community=self.vacine_day*actions

        pop=self.cbg_sizes
        cases=self.cbg_case_num-self.cbg_case_num_old

        vaccine_mat=np.empty((self.num_seed,self.num_cbg))
        for i in range(self.num_community):
            cases_community=cases[:,self.community_index==i]
            cases_community_sum=np.sum(cases_community,axis=1)
            proportion_cases_community=cases_community/cases_community_sum[:,np.newaxis]

            pop_community=pop[self.community_index==i]
            proportion_pop_community=pop_community/np.sum(pop_community)
            proportion_cases_community[cases_community_sum==0,:]=proportion_pop_community

            vaccine_mat[:,self.community_index==i]=vaccine_mat_community[:,i][:,np.newaxis]*proportion_cases_community

        return vaccine_mat.astype(int)

    def SPR_loss_fun(self,input,target):
        input_norm=torch.norm(input,dim=1)
        target_norm=torch.norm(target,dim=1)

        input_normed=input/input_norm.unsqueeze(1)
        target_normed=target/target_norm.unsqueeze(1)

        SPR_loss=-torch.mean(torch.sum(input_normed*target_normed,dim=1))

        return SPR_loss

    def update(self):
        batch_count=0
        critic_loss_sum=0
        actor_loss_sum=0
        SPR_loss_sum=0

        self.buffer_r=(self.buffer_r - self.buffer_r.mean())/(self.buffer_r.std() + 1e-10)

        sa=torch.FloatTensor(self.buffer_sa).to(self.device)
        sadiff=torch.FloatTensor(self.buffer_sadiff).to(self.device).unsqueeze(1)
        r=torch.FloatTensor(self.buffer_r).to(self.device)
        a=torch.FloatTensor(self.buffer_a).to(self.device)
        logp=torch.FloatTensor(self.buffer_logp).to(self.device)
        sa1=torch.FloatTensor(self.buffer_sa1).to(self.device)
        sadiff1=torch.FloatTensor(self.buffer_sadiff1).to(self.device).unsqueeze(1)

        target_v=torch.FloatTensor(self.buffer_pointer,1).to(self.device)
        advantage=torch.FloatTensor(self.buffer_pointer,1).to(self.device)

        with torch.no_grad():
            index_t=np.array(range(self.buffer_pointer))
            batch_pointer_t=0
            batch_num_t=int(self.buffer_pointer/self.batch_size)

            for batch_t in range(batch_num_t):
                batch_index_t=index_t[batch_pointer_t:batch_pointer_t+self.batch_size]
                sa_batch_t=sa[batch_index_t]
                sadiff_batch_t=sadiff[batch_index_t]
                sb_batch_t=self.buffer_sb[batch_index_t]
                r_batch_t=r[batch_index_t]
                sa1_batch_t=sa1[batch_index_t]
                sadiff1_batch_t=sadiff1[batch_index_t]
                sb1_batch_t=self.buffer_sb1[batch_index_t]

                g_batch_t,edge_weight_batch_t,poi_visits_batch_t=self.get_indexed_vectors(sb_batch_t)
                community_encode_batch_t,poi_encode_batch_t=self.online_encoder(sa_batch_t,sadiff_batch_t,self.community_ages_norm_repeated_batch,poi_visits_batch_t,self.poi_areas_norm_repeated_batch,self.poi_times_norm_repeated_batch)
                community_embeddings_batch_t=self.online_GCN(g_batch_t,edge_weight_batch_t,community_encode_batch_t,poi_encode_batch_t)

                g1_batch_t,edge_weight1_batch_t,poi_visits1_batch_t=self.get_indexed_vectors(sb1_batch_t)
                community_encode1_batch_t,poi_encode1_batch_t=self.online_encoder(sa1_batch_t,sadiff1_batch_t,self.community_ages_norm_repeated_batch,poi_visits1_batch_t,self.poi_areas_norm_repeated_batch,self.poi_times_norm_repeated_batch)
                community_embeddings1_batch_t=self.online_GCN(g1_batch_t,edge_weight1_batch_t,community_encode1_batch_t,poi_encode1_batch_t)

                target_v_batch_t=r_batch_t.unsqueeze(1)+self.gamma*self.PPO_critic(community_embeddings1_batch_t)
                advantage_batch_t = (target_v_batch_t - self.PPO_critic(community_embeddings_batch_t))

                target_v[batch_t*self.batch_size:batch_t*self.batch_size+self.batch_size]=target_v_batch_t
                advantage[batch_t*self.batch_size:batch_t*self.batch_size+self.batch_size]=advantage_batch_t

        for _ in range(self.num_epoch):
            index=np.array(range(self.buffer_pointer))
            np.random.shuffle(index)
            batch_pointer=0
            batch_num=int(self.buffer_pointer/self.batch_size)

            for _ in range(batch_num):
                batch_index=index[batch_pointer:batch_pointer+self.batch_size]
                sa_batch=sa[batch_index]
                sadiff_batch=sadiff[batch_index]
                sb_batch=self.buffer_sb[batch_index]
                a_batch=a[batch_index]
                logp_batch=logp[batch_index]
                sa1_batch=sa1[batch_index]
                sadiff1_batch=sadiff1[batch_index]
                sb1_batch=self.buffer_sb1[batch_index]

                target_v_batch=target_v[batch_index]
                advantage_batch=advantage[batch_index]

                g_batch,edge_weight_batch,poi_visits_batch=self.get_indexed_vectors(sb_batch)
                community_encode_batch,poi_encode_batch=self.online_encoder(sa_batch,sadiff_batch,self.community_ages_norm_repeated_batch,poi_visits_batch,self.poi_areas_norm_repeated_batch,self.poi_times_norm_repeated_batch)
                community_embeddings_batch=self.online_GCN(g_batch,edge_weight_batch,community_encode_batch,poi_encode_batch)

                # PPO loss
                mu_batch,sigma_batch=self.PPO_actor(community_embeddings_batch)
                logp_new_batch=self.get_action(mu_batch,sigma_batch,a_batch)
                ratio_batch=torch.exp(logp_new_batch-logp_batch)
                loss_temp1=ratio_batch.unsqueeze(1)*advantage_batch
                loss_temp2=torch.clamp(ratio_batch.unsqueeze(1),1-self.epsilon,1+self.epsilon)*advantage_batch
                actor_loss_batch=torch.mean(torch.min(loss_temp1,loss_temp2))
                actor_loss_sum+=actor_loss_batch.cpu().item()

                v_batch=self.PPO_critic(community_embeddings_batch)
                critic_loss_batch=F.smooth_l1_loss(v_batch,target_v_batch)
                critic_loss_sum+=critic_loss_batch.cpu().item()

                # SPR loss
                community_embeddings_trans_batch=self.transition(community_embeddings_batch,a_batch)
                community_embeddings_proj_online_batch=self.online_projector(community_embeddings_trans_batch)
                community_embeddings_pred_batch=self.predictor(community_embeddings_proj_online_batch)

                with torch.no_grad():
                    g1_batch,edge_weight1_batch,poi_visits1_batch=self.get_indexed_vectors(sb1_batch)
                    community_encode1_target_batch,poi_encode1_target_batch=self.target_encoder(sa1_batch,sadiff1_batch,self.community_ages_norm_repeated_batch,poi_visits1_batch,self.poi_areas_norm_repeated_batch,self.poi_times_norm_repeated_batch)
                    community_embeddings1_target_batch=self.target_GCN(g1_batch,edge_weight1_batch,community_encode1_target_batch,poi_encode1_target_batch)

                    community_embeddings_proj_target_batch=self.target_projector(community_embeddings1_target_batch)

                SPR_loss_batch=self.SPR_loss_fun(community_embeddings_pred_batch,community_embeddings_proj_target_batch)
                SPR_loss_sum+=SPR_loss_batch.cpu().item()

                # update parameters
                loss_batch=self.SPR_weight*SPR_loss_batch+(-actor_loss_batch+critic_loss_batch)
                self.online_encoder_opt.zero_grad()
                self.online_GCN_opt.zero_grad()
                self.transition_opt.zero_grad()
                self.online_projector_opt.zero_grad()
                self.predictor_opt.zero_grad()
                self.PPO_actor_opt.zero_grad()
                self.PPO_critic_opt.zero_grad()

                loss_batch.backward()
                self.online_encoder_opt.step()
                self.online_GCN_opt.step()
                self.transition_opt.step()
                self.online_projector_opt.step()
                self.predictor_opt.step()
                self.PPO_actor_opt.step()
                self.PPO_critic_opt.step()

                for x in self.target_encoder.state_dict().keys():
                    if ('CBGstateNorm.num_batches_tracked' not in x) and ('CBGstatediffNorm.num_batches_tracked' not in x):
                        eval('self.target_encoder.'+x+'.data.mul_(1-self.soft_replace_rate)')
                        eval('self.target_encoder.'+x+'.data.add_(self.soft_replace_rate*self.online_encoder.'+x+'.data)')
                for x in self.target_GCN.state_dict().keys():
                    eval('self.target_GCN.'+x+'.data.mul_(1-self.soft_replace_rate)')
                    eval('self.target_GCN.'+x+'.data.add_(self.soft_replace_rate*self.online_GCN.'+x+'.data)')
                for x in self.target_projector.state_dict().keys():
                    eval('self.target_projector.'+x+'.data.mul_(1-self.soft_replace_rate)')
                    eval('self.target_projector.'+x+'.data.add_(self.soft_replace_rate*self.online_projector.'+x+'.data)')

                batch_pointer+=self.batch_size
                batch_count+=1

        critic_loss_mean=critic_loss_sum/batch_count
        self.critic_loss_trackor.append(critic_loss_mean)
        actor_loss_mean=actor_loss_sum/batch_count
        self.actor_loss_trackor.append(actor_loss_mean)
        SPR_loss_mean=SPR_loss_sum/batch_count
        self.SPR_loss_trackor.append(SPR_loss_mean)

        tqdm.write(f'Update: Critic Loss {critic_loss_mean} | Actor Loss {actor_loss_mean} | SPR Loss {SPR_loss_mean}')

    def save_models(self,episode):
        tqdm.write('Saving models...')
        torch.save(self.online_encoder.state_dict(), os.path.join(self.output_dir,f'online_encoder_{episode}.pth'))
        torch.save(self.target_encoder.state_dict(), os.path.join(self.output_dir,f'target_encoder_{episode}.pth'))
        torch.save(self.online_GCN.state_dict(), os.path.join(self.output_dir,f'online_GCN_{episode}.pth'))
        torch.save(self.target_GCN.state_dict(), os.path.join(self.output_dir,f'target_GCN_{episode}.pth'))
        torch.save(self.transition.state_dict(), os.path.join(self.output_dir,f'transition_{episode}.pth'))
        torch.save(self.online_projector.state_dict(), os.path.join(self.output_dir,f'online_projector_{episode}.pth'))
        torch.save(self.target_projector.state_dict(), os.path.join(self.output_dir,f'target_projector_{episode}.pth'))
        torch.save(self.predictor.state_dict(), os.path.join(self.output_dir,f'predictor_{episode}.pth'))
        torch.save(self.PPO_actor.state_dict(), os.path.join(self.output_dir,f'PPO_actor_{episode}.pth'))
        torch.save(self.PPO_critic.state_dict(), os.path.join(self.output_dir,f'PPO_critic_{episode}.pth'))

        with open(os.path.join(self.output_dir,'episode_cases.json'),'w') as f:
            json.dump(self.episode_cases_trackor,f)
        with open(os.path.join(self.output_dir,'episode_deaths.json'),'w') as f:
            json.dump(self.episode_deaths_trackor,f)
        with open(os.path.join(self.output_dir,'critic_loss.json'),'w') as f:
            json.dump(self.critic_loss_trackor,f)
        with open(os.path.join(self.output_dir,'actor_loss.json'),'w') as f:
            json.dump(self.actor_loss_trackor,f)
        with open(os.path.join(self.output_dir,'SPR_loss.json'),'w') as f:
            json.dump(self.SPR_loss_trackor,f)

    def train(self):
        for episode in tqdm(range(self.num_episode)):
            self.simulator.init_endogenous_variables()
            self.simulator.simulate_disease_spread(length=self.step_length,verbosity=1,no_print=True)
            current_C_community,current_D_community=self.simulator.output_record_community(self.community_index,length=self.step_length)
            self.update_community_state(current_C_community,current_D_community)
            self.community_statediff=self.community_state_num

            self.cbg_case_num=self.simulator.output_last_C()
            self.simulator.empty_record()

            for day in range(1,self.day_length):
                self.buffer_sa[self.buffer_pointer:self.buffer_pointer+self.num_seed]=self.community_state_record
                self.buffer_sadiff[self.buffer_pointer:self.buffer_pointer+self.num_seed]=self.community_statediff
                self.buffer_sb[self.buffer_pointer:self.buffer_pointer+self.num_seed]=np.array([day-1]*self.num_seed)

                with torch.no_grad():
                    community_state_record=torch.FloatTensor(self.community_state_record).to(self.device)
                    community_statediff=torch.FloatTensor(self.community_statediff).to(self.device).unsqueeze(1)
                    index=[day-1]*self.num_seed
                    g,edge_weight,poi_visits=self.get_indexed_vectors(index)

                    community_encode,poi_encode=self.online_encoder(community_state_record,community_statediff,self.community_ages_norm_repeated_seed,poi_visits,self.poi_areas_norm_repeated_seed,self.poi_times_norm_repeated_seed)
                    community_embeddings=self.online_GCN(g,edge_weight,community_encode,poi_encode)
                    mu,sigma=self.PPO_actor(community_embeddings)
                    actions,log_prob=self.get_action(mu,sigma)

                self.buffer_a[self.buffer_pointer:self.buffer_pointer+self.num_seed]=actions.cpu().numpy()
                self.buffer_logp[self.buffer_pointer:self.buffer_pointer+self.num_seed]=log_prob.cpu().numpy()

                vaccine_mat=self.get_vaccine(actions.cpu().numpy())
                self.simulator.add_vaccine(vaccine_mat)
                self.simulator.simulate_disease_spread(length=24,verbosity=1,no_print=True)

                current_C_community,current_D_community=self.simulator.output_record_community(self.community_index,length=self.step_length)
                community_state_old=copy.deepcopy(self.community_state_num)
                self.update_community_state(current_C_community,current_D_community)
                reward=reward_fun(community_state_old,self.community_state_num)
                self.community_statediff=self.community_state_num-community_state_old

                self.cbg_case_num_old=self.cbg_case_num
                self.cbg_case_num=self.simulator.output_last_C()
                self.simulator.empty_record()

                self.buffer_sa1[self.buffer_pointer:self.buffer_pointer+self.num_seed]=self.community_state_record
                self.buffer_sadiff1[self.buffer_pointer:self.buffer_pointer+self.num_seed]=self.community_statediff
                self.buffer_sb1[self.buffer_pointer:self.buffer_pointer+self.num_seed]=np.array([day]*self.num_seed)
                self.buffer_r[self.buffer_pointer:self.buffer_pointer+self.num_seed]=reward

                self.buffer_pointer+=self.num_seed
                if (self.buffer_size-self.buffer_pointer)<self.num_seed:
                    self.update()
                    self.buffer_pointer=0

            episode_C=np.mean(np.sum(self.community_state_num[:,1,:],axis=1),axis=0)
            self.episode_cases_trackor.append(episode_C)
            episode_D=np.mean(np.sum(self.community_state_num[:,2,:],axis=1),axis=0)
            self.episode_deaths_trackor.append(episode_D)

            tqdm.write(f'Episode{episode}: Cases {episode_C} | Deaths {episode_D}')

            if (episode+1)%self.save_interval==0:
                self.save_models(episode+1)

        self.save_models('final')

if __name__ == '__main__':
    import sys
    MSA_name=sys.argv[1]
    batch_size=int(sys.argv[2])

    train_platform=GraphRL(MSA_name=MSA_name,batch_size=batch_size)
    # train_platform.test_simulation()
    # train_platform.test_network()
    train_platform.train()