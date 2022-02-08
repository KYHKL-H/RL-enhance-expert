import numpy as np
import time

class Model:
    def __init__(self,
                 starting_seed=0,
                 num_seeds=1,
                 debug=False,
                 clip_poisson_approximation=True):

        self.starting_seed = starting_seed
        self.num_seeds = num_seeds
        self.debug = debug
        self.clip_poisson_approximation = clip_poisson_approximation

        np.random.seed(self.starting_seed)

    def init_exogenous_variables(self,
                                 poi_areas,
                                 cbg_sizes,
                                 p_sick_at_t0,
                                 poi_psi,
                                 home_beta,
                                 cbg_attack_rates_original,
                                 cbg_death_rates_original,
                                 poi_cbg_visits_list=None,
                                 poi_dwell_time_correction_factors=None,
                                 just_compute_r0=False,
                                 latency_period=96,  # 4 days
                                 infectious_period=84,  # 3.5 days
                                 confirmation_rate=.1,
                                 confirmation_lag=168,  # 7 days
                                 death_lag=432,  # 18 days
                                 no_print=False,
                                 ):
        self.M = len(poi_areas)
        self.N = len(cbg_sizes)
        self.MAX_T=len(poi_cbg_visits_list)

        self.PSI = poi_psi
        self.POI_AREAS = poi_areas
        self.DWELL_TIME_CORRECTION_FACTORS = poi_dwell_time_correction_factors
        self.POI_FACTORS = self.PSI / poi_areas
        if poi_dwell_time_correction_factors is not None:
            self.POI_FACTORS = poi_dwell_time_correction_factors * self.POI_FACTORS
            self.included_dwell_time_correction_factors = True
        else:
            self.included_dwell_time_correction_factors = False
        self.POI_CBG_VISITS_LIST = poi_cbg_visits_list
        self.clipping_monitor = {
        'num_base_infection_rates_clipped':[],
        'num_active_pois':[],
        'num_poi_infection_rates_clipped':[],
        'num_cbgs_active_at_pois':[],
        'num_cbgs_with_clipped_poi_cases':[]}
        # CBG variables
        self.CBG_SIZES = cbg_sizes  
        self.HOME_BETA = home_beta#β0
        self.CBG_ATTACK_RATES_ORIGINAL = cbg_attack_rates_original
        self.CBG_DEATH_RATES_ORIGINAL = cbg_death_rates_original
        self.LATENCY_PERIOD = latency_period
        self.INFECTIOUS_PERIOD = infectious_period
        self.P_SICK_AT_T0 = p_sick_at_t0  # p0

        self.VACCINATION_VECTOR = np.zeros((self.num_seeds,self.N))
        self.VACCINE_ACCEPTANCE = np.ones((self.num_seeds,self.N))
        self.PROTECTION_RATE = 1.0

        self.just_compute_r0 = just_compute_r0
        self.confirmation_rate = confirmation_rate
        self.confirmation_lag = confirmation_lag
        self.death_lag = death_lag

        self.CBG_ATTACK_RATES_NEW = self.CBG_ATTACK_RATES_ORIGINAL * (1-self.PROTECTION_RATE*self.VACCINATION_VECTOR/self.CBG_SIZES)
        self.CBG_DEATH_RATES_NEW = self.CBG_DEATH_RATES_ORIGINAL
        self.CBG_ATTACK_RATES_NEW = np.clip(self.CBG_ATTACK_RATES_NEW, 0, None)
        self.CBG_DEATH_RATES_NEW = np.clip(self.CBG_DEATH_RATES_NEW, 0, None)
        self.CBG_DEATH_RATES_NEW = np.clip(self.CBG_DEATH_RATES_NEW, None, 1)
        assert((self.CBG_DEATH_RATES_NEW>=0).all())
        assert((self.CBG_DEATH_RATES_NEW<=1).all())

    def init_endogenous_variables(self):
        # Initialize exposed/latent individuals
        # Reset
        self.P0 = np.random.binomial(
            self.CBG_SIZES,
            self.P_SICK_AT_T0,
            size=(self.num_seeds, self.N))
        self.cbg_latent = self.P0
        self.cbg_infected = np.zeros((self.num_seeds, self.N))
        self.cbg_removed = np.zeros((self.num_seeds, self.N))
        self.cases_to_confirm = np.zeros((self.num_seeds, self.N))
        self.new_confirmed_cases = np.zeros((self.num_seeds, self.N))
        self.deaths_to_happen = np.zeros((self.num_seeds, self.N))
        self.new_deaths = np.zeros((self.num_seeds, self.N))
        self.C2=np.zeros((self.num_seeds, self.N))
        self.D2=np.zeros((self.num_seeds, self.N))

        self.VACCINATION_VECTOR = np.zeros((self.num_seeds,self.N))

        self.L_1=[]
        self.I_1=[]
        self.R_1=[]
        self.C_1=[]
        self.D_1=[]
        self.T1=[]
        self.t = 0
        self.C=[0]
        self.D=[0]
        self.history_C2 = []
        self.history_D2 = []
        self.epidemic_over = False

    def reset_random_seed(self):
        np.random.seed(self.starting_seed)

    def add_vaccine(self,vaccine_vector):
        self.VACCINATION_VECTOR+=vaccine_vector
        self.VACCINATION_VECTOR = np.clip(self.VACCINATION_VECTOR, None, (self.CBG_SIZES*self.VACCINE_ACCEPTANCE))
        self.CBG_ATTACK_RATES_NEW = self.CBG_ATTACK_RATES_ORIGINAL * (1-self.PROTECTION_RATE*self.VACCINATION_VECTOR/self.CBG_SIZES)
        self.CBG_ATTACK_RATES_NEW = np.clip(self.CBG_ATTACK_RATES_NEW, 0, None)

    def get_new_infectious(self):
        new_infectious = np.random.binomial(self.cbg_latent.astype(int), 1 / self.LATENCY_PERIOD)
        return new_infectious

    def get_new_removed(self):
        new_removed = np.random.binomial(self.cbg_infected.astype(int), 1 / self.INFECTIOUS_PERIOD)
        return new_removed

    def format_floats(self, arr):
        return [int(round(x)) for x in arr]

    def simulate_disease_spread(self,length=24,verbosity=1,no_print=False):
        assert(self.t<self.MAX_T)
        t_start=self.t
        time_start=time.time()

        while self.t-t_start < length:
            iter_t0 = time.time()
            if (verbosity > 0) and (self.t % verbosity == 0):
                L = np.sum(self.cbg_latent, axis=1)
                I = np.sum(self.cbg_infected, axis=1)
                R = np.sum(self.cbg_removed, axis=1)

                self.T1.append(self.t)
                self.L_1.append(L)
                self.I_1.append(I)
                self.R_1.append(R)
                self.C_1.append(self.C)
                self.D_1.append(self.D)

                self.history_C2.append(self.C2) # Save history for cases
                self.history_D2.append(self.D2) # Save history for deaths

                if(no_print==False):
                    print('t:',self.t,'L:',L,'I:',I,'R',R,'C',self.C,'D',self.D)

            self.update_states(self.t)
            C1 = np.sum(self.new_confirmed_cases,axis=1)
            self.C2=self.C2+self.new_confirmed_cases
            self.C[0]=self.C[0]+C1
            D1 = np.sum(self.new_deaths,axis=1)
            self.D2=self.D2+self.new_deaths
            self.D[0]=self.D[0]+D1
            if self.debug and verbosity > 0 and self.t % verbosity == 0:
                print('Num active POIs: %d. Num with infection rates clipped: %d' % (self.num_active_pois, self.num_poi_infection_rates_clipped))
                print('Num CBGs active at POIs: %d. Num with clipped num cases from POIs: %d' % (self.num_cbgs_active_at_pois, self.num_cbgs_with_clipped_poi_cases))
            if self.debug:
                print("Time for iteration %i: %2.3f seconds" % (self.t, time.time() - iter_t0))

            self.t += 1

        # print(f'Simulate {length} steps in {time.time()-time_start}s')

    def empty_record(self):
        self.L_1=[]
        self.I_1=[]
        self.R_1=[]
        self.C_1=[]
        self.D_1=[]
        self.T1=[]
        self.history_C2 = []
        self.history_D2 = []

    def output_record(self,full=False,length=24):
        cbg_all_affected = self.cbg_latent + self.cbg_infected + self.cbg_removed
        total_affected = np.sum(cbg_all_affected, axis=1)

        '''
        T1                  T1
        L_1                 T1*S
        I_1                 T1*S
        R_1                 T1*S
        self.C2             S*N
        self.D2             S*N
        history_C2          T1*S*N
        history_D2          T1*S*N
        total_affected      S
        cbg_all_affected    S*N

        history_C2          S*T1*N
        history_D2          S*T1*N
        '''

        if full:
            print('Output records')
            o_T1=np.array(self.T1)
            o_L_1=np.array(self.L_1)
            o_I_1=np.array(self.I_1)
            o_R_1=np.array(self.R_1)
            o_C2=np.array(self.C2)
            o_D2=np.array(self.D2)
            o_history_C2=np.array(self.history_C2)
            o_history_D2=np.array(self.history_D2)
            o_total_affected=np.array(total_affected)
            o_cbg_all_affected=np.array(cbg_all_affected)

            o_history_C2=np.transpose(o_history_C2,(1,0,2))
            o_history_D2=np.transpose(o_history_D2,(1,0,2))

            return o_T1,o_L_1,o_I_1,o_R_1,o_C2,o_D2, o_total_affected, o_history_C2, o_history_D2, o_cbg_all_affected

        else:
            o_history_C2=np.array(self.history_C2)[-length:,:,:]
            o_history_D2=np.array(self.history_D2)[-length:,:,:]

            o_history_C2=np.transpose(o_history_C2,(1,0,2))
            o_history_D2=np.transpose(o_history_D2,(1,0,2))

            return o_history_C2,o_history_D2

    def output_record_community(self,community_index,length=24):
        o_history_C2=np.array(self.history_C2)[-length:,:,:]
        o_history_D2=np.array(self.history_D2)[-length:,:,:]

        o_history_C2=np.transpose(o_history_C2,(1,0,2))
        o_history_D2=np.transpose(o_history_D2,(1,0,2))

        num_community=np.max(community_index)+1

        C_community=np.empty((o_history_C2.shape[0],o_history_C2.shape[1],num_community))
        D_community=np.empty((o_history_D2.shape[0],o_history_D2.shape[1],num_community))

        for i in range(num_community):
            C_community[:,:,i]=np.sum(o_history_C2[:,:,community_index==i],axis=-1)
            D_community[:,:,i]=np.sum(o_history_D2[:,:,community_index==i],axis=-1)

        return C_community,D_community

    def output_last_C(self):
        o_history_C2=np.array(self.history_C2)[-1,:,:]

        return o_history_C2

    def update_states(self, t):
        '''
        Applies one round of updates. First, we compute the infection rates
        at each POI depending on which CBGs are visiting it at time t. Based
        on the home and POI infection rates, we compute the number of new
        cases per CBG. Then, we update the SLIR states accordingly.
        '''
        self.get_new_cases(t)
        new_infectious = self.get_new_infectious()
        new_removed = self.get_new_removed()
        if not self.just_compute_r0:
            self.cbg_latent = self.cbg_latent + self.cbg_new_cases - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed
            self.new_confirmed_cases = np.random.binomial(self.cases_to_confirm.astype(int), 1/self.confirmation_lag)
            new_cases_to_confirm = np.random.binomial(new_infectious.astype(int), self.confirmation_rate)
            self.cases_to_confirm = self.cases_to_confirm + new_cases_to_confirm - self.new_confirmed_cases

            self.new_deaths = np.random.binomial(self.deaths_to_happen.astype(int), 1/self.death_lag)
            new_deaths_to_happen = np.random.binomial(new_infectious.astype(int), self.CBG_DEATH_RATES_NEW)
            self.deaths_to_happen = self.deaths_to_happen + new_deaths_to_happen - self.new_deaths
        else:
            self.cbg_latent = self.cbg_latent - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed + self.cbg_new_cases

    def get_new_cases(self, t):
        '''
        Determines the number of new cases per CBG. This depends on the CBG's
        home infection rate and the infection rates of the POIs that members
        from this CBG visited at time t. If the model is stochastic, the
        number of new cases is drawn randomly; otherwise, the expectation of the
        random variable is used.

        This method computes the weighted rates then uses a Poisson approximation.
        '''
        # M is number of POIs
        # N is number of CBGs
        # S is number of seeds

        ### Compute CBG densities and infection rates
        cbg_densities = self.cbg_infected / self.CBG_SIZES
        overall_densities = (np.sum(self.cbg_infected, axis=1) / np.sum(self.CBG_SIZES)).reshape(-1, 1)
        num_sus = np.clip(self.CBG_SIZES - self.cbg_latent - self.cbg_infected - self.cbg_removed, 0, None)
        sus_frac = num_sus / self.CBG_SIZES

        if self.PSI > 0:
            # Our model: can only be infected by people in your home CBG.
            #cbg_base_infection_rates = self.HOME_BETA * cbg_densities
            cbg_base_infection_rates = self.HOME_BETA * self.CBG_ATTACK_RATES_NEW * cbg_densities
            cbg_base_infection_rates=np.nan_to_num(cbg_base_infection_rates)
        else:
            cbg_base_infection_rates = np.tile(overall_densities, self.N) * self.HOME_BETA  # S x N
        self.num_base_infection_rates_clipped = np.sum(cbg_base_infection_rates > 1)
        cbg_base_infection_rates = np.clip(cbg_base_infection_rates, None, 1.0)

        ### Load or compute POI x CBG matrix
        if self.POI_CBG_VISITS_LIST is not None:  # try to load
            poi_cbg_visits = self.POI_CBG_VISITS_LIST[t]
            poi_visits = poi_cbg_visits @ np.ones(poi_cbg_visits.shape[1])

        if not self.just_compute_r0:
          # use network data
            self.num_active_pois = np.sum(poi_visits > 0)
            col_sums = np.squeeze(np.array(poi_cbg_visits.sum(axis=0)))
            self.cbg_num_out = col_sums
            poi_infection_rates = self.POI_FACTORS * (poi_cbg_visits @ cbg_densities.T).T
            self.num_poi_infection_rates_clipped = np.sum(poi_infection_rates > 1)
            if self.clip_poisson_approximation:
                poi_infection_rates = np.clip(poi_infection_rates, None, 1.0)

            cbg_mean_new_cases_from_poi = self.CBG_ATTACK_RATES_NEW * sus_frac * (poi_infection_rates @ poi_cbg_visits)
            cbg_mean_new_cases_from_poi=np.nan_to_num(cbg_mean_new_cases_from_poi)
            cbg_mean_new_cases_from_poi = cbg_mean_new_cases_from_poi.astype(np.float64)
            num_cases_from_poi = np.random.poisson(cbg_mean_new_cases_from_poi)
            self.num_cbgs_active_at_pois = np.sum(cbg_mean_new_cases_from_poi > 0)

        self.num_cbgs_with_clipped_poi_cases = np.sum(num_cases_from_poi > num_sus)
        self.cbg_new_cases_from_poi = np.clip(num_cases_from_poi, None, num_sus)
        num_sus_remaining = num_sus - self.cbg_new_cases_from_poi

        self.cbg_new_cases_from_base = np.random.binomial(
            num_sus_remaining.astype(int),
            cbg_base_infection_rates)
        self.cbg_new_cases = self.cbg_new_cases_from_poi + self.cbg_new_cases_from_base

        self.clipping_monitor['num_base_infection_rates_clipped'].append(self.num_base_infection_rates_clipped)
        self.clipping_monitor['num_active_pois'].append(self.num_active_pois)
        self.clipping_monitor['num_poi_infection_rates_clipped'].append(self.num_poi_infection_rates_clipped)
        self.clipping_monitor['num_cbgs_active_at_pois'].append(self.num_cbgs_active_at_pois)
        self.clipping_monitor['num_cbgs_with_clipped_poi_cases'].append(self.num_cbgs_with_clipped_poi_cases)
        assert (self.cbg_new_cases <= num_sus).all()