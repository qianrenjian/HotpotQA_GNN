import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features=768, out_features=32, dropout=0.2, alpha=0.3, concat=True, nodes_num=200):
        # features=dim, hidden=8
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.randn(size=(in_features, out_features))) # (dim, 8)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.randn(size=(2*out_features, 1))) #(2*8,1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.normal_layer = nn.BatchNorm1d(nodes_num)

    def forward(self, feat_matrix, adj):
        # features (B, N, dim) , adj (B, N, N)
        print(feat_matrix.device, self.W.device)
        h = torch.matmul(feat_matrix, self.W) # (B,N,8)
                
        N = h.shape[-2] # N
        B = feat_matrix.shape[0]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=-1)\
                                        .view(-1, N, N, 2 * self.out_features) # (B, N, N, 16)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1)) # (B, N, N, 16) * (16, 1) --> (B, N, N)
        
        minimal_vec = -9e9*torch.ones_like(e) # (B, N, N) 为了softmax时候必定最小. 不取0是避免分母为0.
        attention = torch.where(adj > 0, e, minimal_vec) # 都是[B, N, N]
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = self.normal_layer(torch.bmm(attention, h))  # (B, N, N)*(B, N ,out_features)        

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime # [B, N, 8]

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT_HotpotQA(nn.Module):
    def __init__(self, features=768, hidden=32, nclass=2, dropout=0.2, alpha=0.3, nheads=8, nodes_num=200):
        super(GAT_HotpotQA, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(features, hidden, dropout=dropout, 
                                               alpha=alpha, concat=True, nodes_num=nodes_num) \
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # hidden * nheads = 8*8, nclass= 7 
        self.out_att_sent = GraphAttentionLayer(hidden * nheads, nclass, 
                                                dropout=dropout, alpha=alpha, concat=False, nodes_num=nodes_num)
        
        self.out_att_para = GraphAttentionLayer(hidden * nheads, nclass, 
                                                dropout=dropout, alpha=alpha, concat=False, nodes_num=nodes_num)
        
        self.out_att_Qtype = GraphAttentionLayer(hidden * nheads, hidden, 
                                                 dropout=dropout, alpha=alpha, concat=False, nodes_num=nodes_num)
        
        self.W2 = nn.Parameter(torch.randn(size=(hidden, 2)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        
        self.normal_layer = nn.BatchNorm1d(nodes_num) # 200 Node
        

    def forward(self, feat_matrix, adj):
        print(f"self.W2.device: {self.W2.device}")
        feat_matrix = feat_matrix.to(self.W2.device)
        adj = adj.to(self.W2.device)
        # features (B, N, dim) , adj (B, N, N)
        feat_matrix = F.dropout(feat_matrix, self.dropout, training=self.training)
        feat_matrix = torch.cat([att(feat_matrix, adj) for att in self.attentions], dim=-1) # (B,N,hidden*heads)
        feat_matrix = F.dropout(self.normal_layer(feat_matrix), self.dropout, training=self.training)
        
        logits_sent = torch.sigmoid(self.out_att_sent(feat_matrix, adj))
        logits_para = F.elu(self.out_att_para(feat_matrix, adj))
        logits_Qtype = F.elu(torch.matmul(self.out_att_Qtype(feat_matrix, adj)[:,0,:], self.W2)).view(-1,2)
        
        return logits_sent, logits_para, logits_Qtype # 前2个:[B, N, num_class] 最后:[B,2]

if __name__ == '__main__':
    model = GAT_HotpotQA()
    print(model.W2.device)
