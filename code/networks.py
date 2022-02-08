import torch
from torch import nn
import torch.nn.functional as F
import dgl.nn as gnn

Encode_CBGstate_hidden_dim1=32
Encode_CBGstate_hidden_dim2=16
Encode_CBGstatediff_hidden_dim1=32
Encode_CBGstatediff_hidden_dim2=16
Encode_CBGages_hidden_dim1=32
Encode_CBGages_hidden_dim2=16
Encode_CBGcat_hidden_dim1=32
Encode_CBGcat_hidden_dim2=16
Encode_POIvisit_hidden_dim1=4
Encode_POIvisit_hidden_dim2=8
Encode_POIareas_hidden_dim1=4
Encode_POIareas_hidden_dim2=8
Encode_POItimes_hidden_dim1=4
Encode_POItimes_hidden_dim2=8
Encode_POIcat_hidden_dim1=16
Encode_POIcat_hidden_dim2=16
Encode_output_dim=8

Embedding_hidden_dim=16
Embedding_dim=8

Transition_Embed_hidden_dim_1=16
Transition_Embed_hidden_dim_2=32
Transition_Action_hidden_dim_1=8
Transition_Action_hidden_dim_2=16
Transition_cat_hidden_dim_1=16

PPO_hidden_dim1=128
PPO_hidden_dim2=64

Actor_mu_hidden_dim=64
Actor_sigma_hidden_dim=64
Critic_hidden_dim=16

Projector_dim=16

Predictor_hidden_dim1=32
Predictor_hidden_dim2=32

class Encoder(nn.Module):
    def __init__(self,step_length):
        super(Encoder,self).__init__()
        self.step_length=step_length

        self.CBGstateNorm=nn.BatchNorm2d(num_features=3)
        self.CBGstatediffNorm=nn.BatchNorm2d(num_features=3)

        self.CBGstateLin1=nn.Linear(in_features=3*self.step_length,out_features=Encode_CBGstate_hidden_dim1)
        self.CBGstateLin2=nn.Linear(in_features=Encode_CBGstate_hidden_dim1,out_features=Encode_CBGstate_hidden_dim2)
        self.CBGstatediffLin1=nn.Linear(in_features=3,out_features=Encode_CBGstatediff_hidden_dim1)
        self.CBGstatediffLin2=nn.Linear(in_features=Encode_CBGstatediff_hidden_dim1,out_features=Encode_CBGstatediff_hidden_dim2)

        self.CBGagesLin1=nn.Linear(in_features=23,out_features=Encode_CBGages_hidden_dim1)
        self.CBGagesLin2=nn.Linear(in_features=Encode_CBGages_hidden_dim1,out_features=Encode_CBGages_hidden_dim2)

        self.CBGcatLin1=nn.Linear(in_features=Encode_CBGstate_hidden_dim2+Encode_CBGstatediff_hidden_dim2+Encode_CBGages_hidden_dim2,out_features=Encode_CBGcat_hidden_dim1)
        self.CBGcatLin2=nn.Linear(in_features=Encode_CBGcat_hidden_dim1,out_features=Encode_CBGcat_hidden_dim2)
        self.CBGoutLin=nn.Linear(in_features=Encode_CBGcat_hidden_dim2,out_features=Encode_output_dim)

        self.POIvisitLin1=nn.Linear(in_features=1,out_features=Encode_POIvisit_hidden_dim1)
        self.POIvisitLin2=nn.Linear(in_features=Encode_POIvisit_hidden_dim1,out_features=Encode_POIvisit_hidden_dim2)
        self.POIareasLin1=nn.Linear(in_features=1,out_features=Encode_POIareas_hidden_dim1)
        self.POIareasLin2=nn.Linear(in_features=Encode_POIareas_hidden_dim1,out_features=Encode_POIareas_hidden_dim2)
        self.POItimesLin1=nn.Linear(in_features=1,out_features=Encode_POItimes_hidden_dim1)
        self.POItimesLin2=nn.Linear(in_features=Encode_POItimes_hidden_dim1,out_features=Encode_POItimes_hidden_dim2)

        self.POIcatLin1=nn.Linear(in_features=Encode_POIvisit_hidden_dim2+Encode_POIareas_hidden_dim2+Encode_POItimes_hidden_dim2,out_features=Encode_POIcat_hidden_dim1)
        self.POIcatLin2=nn.Linear(in_features=Encode_POIcat_hidden_dim1,out_features=Encode_POIcat_hidden_dim2)
        self.POIoutLin=nn.Linear(in_features=Encode_POIcat_hidden_dim2,out_features=Encode_output_dim)

    def forward(self,cbg_state,cbg_statediff,cbg_ages,poi_visit,poi_areas,poi_times):
        batch_size=len(cbg_state)

        # cbg_state     S*T*3*N
        cbg_state=cbg_state.permute(0,2,1,3)# 转为S*3*T*N
        cbg_state=self.CBGstateNorm(cbg_state)# S为batch_size，3为channel，T*N看做W*H
        cbg_state=cbg_state.permute(0,3,1,2)# 转为S*N*3*T
        cbg_state=cbg_state.reshape(batch_size,-1,3*self.step_length)

        # cbg_state(normed)     S*N*(3*T)
        cbg_state=F.leaky_relu(self.CBGstateLin1(cbg_state))
        cbg_state=F.leaky_relu(self.CBGstateLin2(cbg_state))

        # cbg_statediff     S*1*3*N
        cbg_statediff=cbg_statediff.permute(0,2,1,3)# 转为S*3*1*N
        cbg_statediff=self.CBGstatediffNorm(cbg_statediff)# S为batch_size，3为channel，1*N看做W*H
        cbg_statediff=cbg_statediff.permute(0,3,1,2)# 转为S*N*3*1
        cbg_statediff=cbg_statediff.reshape(batch_size,-1,3)

        # cbg_statediff(normed)     S*N*(3*T)
        cbg_statediff=F.leaky_relu(self.CBGstatediffLin1(cbg_statediff))
        cbg_statediff=F.leaky_relu(self.CBGstatediffLin2(cbg_statediff))

        cbg_ages=F.leaky_relu(self.CBGagesLin1(cbg_ages))
        cbg_ages=F.leaky_relu(self.CBGagesLin2(cbg_ages))

        cbg_cat=torch.cat((cbg_state,cbg_statediff,cbg_ages),dim=2)
        cbg_cat=F.leaky_relu(self.CBGcatLin1(cbg_cat))
        cbg_cat=F.leaky_relu(self.CBGcatLin2(cbg_cat))
        cbg_encode=self.CBGoutLin(cbg_cat)

        poi_visit=F.leaky_relu(self.POIvisitLin1(poi_visit))
        poi_visit=F.leaky_relu(self.POIvisitLin2(poi_visit))

        poi_areas=F.leaky_relu(self.POIareasLin1(poi_areas))
        poi_areas=F.leaky_relu(self.POIareasLin2(poi_areas))

        poi_times=F.leaky_relu(self.POItimesLin1(poi_times))
        poi_times=F.leaky_relu(self.POItimesLin2(poi_times))

        poi_cat=torch.cat((poi_visit,poi_areas,poi_times),dim=2)
        poi_cat=F.leaky_relu(self.POIcatLin1(poi_cat))
        poi_cat=F.leaky_relu(self.POIcatLin2(poi_cat))
        poi_encode=self.POIoutLin(poi_cat)

        return cbg_encode,poi_encode

class GCN(nn.Module):
    def __init__(self):
        super(GCN,self).__init__()

        self.GraphConv1=gnn.GraphConv(in_feats=Encode_output_dim, out_feats=Embedding_hidden_dim, norm='both', weight=True,allow_zero_in_degree=True)
        self.GraphConv2=gnn.GraphConv(in_feats=Embedding_hidden_dim, out_feats=Embedding_dim, norm='both', weight=True,allow_zero_in_degree=True)

    def forward(self,g,edge_weight,cbg_encode,poi_encode):
        # cbg_encode    batch_size*cbg*Encode_output_dim
        # poi_encode    batch_size*poi*Encode_output_dim
        batch_size=len(cbg_encode)
        cbg_num=cbg_encode.shape[1]

        feature=torch.cat((cbg_encode,poi_encode),dim=1)
        embeddings=list()
        for i in range(batch_size):
            h=F.leaky_relu(self.GraphConv1(g[i],feature[i],edge_weight=edge_weight[i]))
            h=self.GraphConv2(g[i],h,edge_weight=edge_weight[i]).unsqueeze(0)
            embeddings.append(h)

        embeddings=torch.cat(embeddings,dim=0)
        cbg_embeddings=embeddings[:,:cbg_num,:]

        return cbg_embeddings

class Transition(nn.Module):
    def __init__(self):
        super(Transition,self).__init__()

        self.EmbedLin1=nn.Linear(in_features=Embedding_dim,out_features=Transition_Embed_hidden_dim_1)
        self.EmbedLin2=nn.Linear(in_features=Transition_Embed_hidden_dim_1,out_features=Transition_Embed_hidden_dim_2)

        self.ActionLin1=nn.Linear(in_features=1,out_features=Transition_Action_hidden_dim_1)
        self.ActionLin2=nn.Linear(in_features=Transition_Action_hidden_dim_1,out_features=Transition_Action_hidden_dim_2)

        self.Lin1=nn.Linear(in_features=Transition_Embed_hidden_dim_2+Transition_Action_hidden_dim_2,out_features=Transition_cat_hidden_dim_1)
        self.Lin2=nn.Linear(in_features=Transition_cat_hidden_dim_1,out_features=Embedding_dim)

    def forward(self,cbg_embeddings,action):
        cbg_embeddings=F.leaky_relu(self.EmbedLin1(cbg_embeddings))
        cbg_embeddings=F.leaky_relu(self.EmbedLin2(cbg_embeddings))

        action=action.unsqueeze(-1)
        action=F.leaky_relu(self.ActionLin1(action))
        action=F.leaky_relu(self.ActionLin2(action))

        cbg_action_cat=torch.cat((cbg_embeddings,action),dim=2)
        cbg_action_cat=F.leaky_relu(self.Lin1(cbg_action_cat))
        cbg_embeddings_new=self.Lin2(cbg_action_cat)

        return cbg_embeddings_new

class Actor(nn.Module):
    def __init__(self,num_cbg):
        super(Actor,self).__init__()

        self.num_cbg=num_cbg
        self.Lin1=nn.Linear(in_features=self.num_cbg*Embedding_dim,out_features=PPO_hidden_dim1)
        self.Lin2=nn.Linear(in_features=PPO_hidden_dim1,out_features=PPO_hidden_dim2)

        self.mu_Lin1=nn.Linear(in_features=PPO_hidden_dim2,out_features=Actor_mu_hidden_dim)
        self.mu_Lin2=nn.Linear(in_features=Actor_mu_hidden_dim,out_features=self.num_cbg)

        self.sigma_Lin1=nn.Linear(in_features=PPO_hidden_dim2,out_features=Actor_sigma_hidden_dim)
        self.sigma_Lin2=nn.Linear(in_features=Actor_sigma_hidden_dim,out_features=self.num_cbg)

    def forward(self,cbg_embeddings):
        batch_size=len(cbg_embeddings)
        cbg_embeddings=cbg_embeddings.reshape(batch_size,self.num_cbg*Embedding_dim)

        cbg_embeddings=F.leaky_relu(self.Lin1(cbg_embeddings))
        cbg_embeddings=F.leaky_relu(self.Lin2(cbg_embeddings))

        mu_hidden=F.leaky_relu(self.mu_Lin1(cbg_embeddings))
        mu=self.mu_Lin2(mu_hidden)

        sigma_hidden=F.leaky_relu(self.sigma_Lin1(cbg_embeddings))
        sigma=self.sigma_Lin2(sigma_hidden)

        return mu,sigma

class Critic(nn.Module):
    def __init__(self,num_cbg):
        super(Critic,self).__init__()

        self.num_cbg=num_cbg
        self.Lin1=nn.Linear(in_features=self.num_cbg*Embedding_dim,out_features=PPO_hidden_dim1)
        self.Lin2=nn.Linear(in_features=PPO_hidden_dim1,out_features=PPO_hidden_dim2)
        self.Lin3=nn.Linear(in_features=PPO_hidden_dim2,out_features=Critic_hidden_dim)
        self.Lin4=nn.Linear(in_features=Critic_hidden_dim,out_features=1)

    def forward(self,cbg_embeddings):
        batch_size=len(cbg_embeddings)
        cbg_embeddings=cbg_embeddings.reshape(batch_size,self.num_cbg*Embedding_dim)

        cbg_embeddings=F.leaky_relu(self.Lin1(cbg_embeddings))
        cbg_embeddings=F.leaky_relu(self.Lin2(cbg_embeddings))
        cbg_embeddings=F.leaky_relu(self.Lin3(cbg_embeddings))
        value=self.Lin4(cbg_embeddings)

        return  value

class Projector(nn.Module):
    def __init__(self,num_cbg):
        super(Projector,self).__init__()

        self.num_cbg=num_cbg
        self.Lin1=nn.Linear(in_features=self.num_cbg*Embedding_dim,out_features=PPO_hidden_dim1)
        self.Lin2=nn.Linear(in_features=PPO_hidden_dim1,out_features=PPO_hidden_dim2)
        self.Lin3=nn.Linear(in_features=PPO_hidden_dim2,out_features=Projector_dim)

    def forward(self,cbg_embeddings):
        batch_size=len(cbg_embeddings)
        cbg_embeddings=cbg_embeddings.reshape(batch_size,self.num_cbg*Embedding_dim)

        cbg_embeddings=F.leaky_relu(self.Lin1(cbg_embeddings))
        cbg_embeddings=F.leaky_relu(self.Lin2(cbg_embeddings))
        Projection=self.Lin3(cbg_embeddings)

        return Projection

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor,self).__init__()

        self.Lin1=nn.Linear(in_features=Projector_dim,out_features=Predictor_hidden_dim1)
        self.Lin2=nn.Linear(in_features=Predictor_hidden_dim1,out_features=Predictor_hidden_dim2)
        self.Lin3=nn.Linear(in_features=Predictor_hidden_dim2,out_features=Projector_dim)

    def forward(self,Projection):

        Projection=F.leaky_relu(self.Lin1(Projection))
        Projection=F.leaky_relu(self.Lin2(Projection))
        Projection_pred=self.Lin3(Projection)

        return Projection_pred