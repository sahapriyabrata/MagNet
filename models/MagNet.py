import torch
import torch.nn as nn

class MagNet(nn.Module):

    def __init__(self, num_objects, in_size=4, hidden_size=64, int_feature_size=8, self_feature_size=4, out_size=2):
        super(MagNet, self).__init__()
        self.num_objects = num_objects
        self.n_pairs = num_objects*(num_objects-1)
        self.out_size = out_size
        self.int_feature_size_per_dim = int_feature_size//out_size
        self.self_feature_size_per_dim = self_feature_size//out_size
        self.L1 = nn.Linear(in_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, hidden_size)
        self.I1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.I2 = nn.Linear(hidden_size, int_feature_size, bias=False)
        self.I3 = torch.empty(self.n_pairs, out_size, self.int_feature_size_per_dim, 1, requires_grad=True)
        self.S1 = nn.Linear(in_size, self_feature_size)
        self.S2W = torch.empty(self.num_objects, out_size, self.self_feature_size_per_dim, 1, requires_grad=True)
        self.S2b = torch.zeros(self.num_objects, out_size, requires_grad=True)
        self.Table = self.make_table()
        nn.init.normal_(self.I3, 0, 0.01)
        nn.init.normal_(self.S2W, 0, 0.01)


    def make_table(self):
        Table = torch.zeros((self.num_objects, self.num_objects), dtype=torch.int)
        count = 0
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if i == j:
                    continue
                Table[i, j] = count
                count += 1
        return Table


    def forward(self, inputs):
        seq_len, num_objects = inputs.shape[:2]
        predictions = []
        hidden_states_int = self.L1(inputs)
        hidden_states_int = torch.relu(hidden_states_int)
        hidden_states_int = self.L2(hidden_states_int)
        hidden_states_int = torch.relu(hidden_states_int)
        hidden_states_self = self.S1(inputs)
        hidden_states_self = torch.relu(hidden_states_self)
        for i in range(num_objects):
            interaction = torch.zeros(seq_len, self.out_size)
            for j in range(num_objects):
                if i == j:
                    continue
                Xi = hidden_states_int[:,i,:]
                Xj = hidden_states_int[:,j,:]
                pairwise = self.I1(Xi-Xj)
                pairwise = torch.tanh(pairwise)
                pairwise = self.I2(pairwise)
                pairwise = torch.tanh(pairwise)
                for dim in range(self.out_size):
                    interaction[:,dim:dim+1] += torch.matmul(pairwise[:,dim*self.int_feature_size_per_dim:(dim+1)*self.int_feature_size_per_dim],
                                                       self.I3[self.Table[i, j], dim])
            prediction = interaction
            Si = hidden_states_self[:,i,:]
            for dim in range(self.out_size):
                prediction[:,dim:dim+1] += torch.matmul(Si[:,dim*self.self_feature_size_per_dim:(dim+1)*self.self_feature_size_per_dim],
                                                  self.S2W[i, dim]) + self.S2b[i, dim]
            predictions.append(prediction)
        predictions = torch.stack(predictions)
        predictions = torch.transpose(predictions, 1, 0)

        return predictions
