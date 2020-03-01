import torch
import torch.nn as nn

class MagNet(nn.Module):

    def __init__(self, n_objects, in_size=4, hidden_size=256, int_feature_size=32, self_feature_size=32, out_size=2):
        super(MagNet, self).__init__()
        self.n_objects = n_objects
        self.n_pairs = n_objects*(n_objects-1)
        self.out_size = out_size
        self.int_feature_size_per_dim = int_feature_size//out_size
        self.self_feature_size_per_dim = self_feature_size//out_size
        self.L1 = nn.Linear(in_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, hidden_size)
        self.I1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.I2 = nn.Linear(hidden_size, int_feature_size, bias=False)
        self.I3 = torch.empty(self.n_pairs, out_size, self.int_feature_size_per_dim, 1, requires_grad=True)
        self.S0 = nn.Linear(in_size, hidden_size)
        self.S1 = nn.Linear(hidden_size, self_feature_size)
        self.S2W = torch.empty(self.n_objects, out_size, self.self_feature_size_per_dim, 1, requires_grad=True)
        self.S2b = torch.zeros(self.n_objects, out_size, requires_grad=True)
        self.DM = self.difference_matrix()
        nn.init.normal_(self.I3, 0, 0.01)
        nn.init.normal_(self.S2W, 0, 0.01)

    def difference_matrix(self):
        DM = torch.zeros((self.n_objects, self.n_pairs))
        count = 0
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if i == j:
                    continue
                DM[i,count] = 1
                DM[j,count] = -1
                count += 1
        return DM        

    def forward(self, inputs):
        seq_len, n_objects = inputs.shape[:2]
        predictions = []
        hidden_states_int = self.L1(inputs)
        hidden_states_int = torch.relu(hidden_states_int)
        hidden_states_int = self.L2(hidden_states_int)
        hidden_states_int = torch.relu(hidden_states_int)
        hidden_states_self = self.S0(inputs)
        hidden_states_self = torch.relu(hidden_states_self)
        hidden_states_self = self.S1(hidden_states_self)
        hidden_states_self = torch.relu(hidden_states_self)

        pairwise = torch.matmul(torch.transpose(hidden_states_int, 1, 2), self.DM)
        pairwise = torch.transpose(pairwise, 1, 2)
        pairwise = self.I1(pairwise)
        pairwise = torch.tanh(pairwise)
        pairwise = self.I2(pairwise)
        pairwise = torch.tanh(pairwise)
        int_comp_pairwise = torch.zeros(seq_len, self.n_pairs, self.out_size, 1)
        for dim in range(self.out_size):
            features = pairwise[:,:,dim*self.int_feature_size_per_dim:(dim+1)*self.int_feature_size_per_dim].unsqueeze(-2)
            weights = self.I3[:,dim].unsqueeze(0)
            int_comp_pairwise[:,:,dim:dim+1] = torch.matmul(features, weights)   
        int_comp_pairwise = int_comp_pairwise.squeeze(-1)
        SM = self.DM.clone().t()
        SM[SM == -1] = 0
        int_comp = torch.matmul(torch.transpose(int_comp_pairwise, 1, 2), SM)
        int_comp = torch.transpose(int_comp, 1, 2) 
        
        self_comp = torch.zeros(seq_len, n_objects, self.out_size, 1)
        for dim in range(self.out_size):
            features = hidden_states_self[:,:,dim*self.self_feature_size_per_dim:(dim+1)*self.self_feature_size_per_dim].unsqueeze(-2)
            weights = self.S2W[:,dim].unsqueeze(0)
            bias = self.S2b[:,dim:dim+1].unsqueeze(0).unsqueeze(-1)
            self_comp[:,:,dim:dim+1] = torch.matmul(features, weights) + bias

        predictions = int_comp + self_comp.squeeze(-1)

        return predictions
