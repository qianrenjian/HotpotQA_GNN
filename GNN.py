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

    def forward(self, feat_matrix, adj, index=-1):
        # features (B, N, dim) , adj (B, N, N)
        h = torch.matmul(feat_matrix, self.W) # (B,N,8)
                
        N = h.shape[-2] # N
        B = feat_matrix.shape[0]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=-1)\
                                        .view(-1, N, N, 2 * self.out_features) # (B, N, N, 16)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1)) # (B, N, N, 16) * (16, 1) --> (B, N, N)
        
        if e.dtype==torch.float16:
            minimal_vec = -65504*torch.ones_like(e) # (B, N, N) 为了softmax时候必定最小. 不取0是避免分母为0.
        else:
            minimal_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, minimal_vec) # 都是[B, N, N]
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)  # (B, N, N)*(B, N ,out_features)        

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

        self.attentions = nn.ModuleList([GraphAttentionLayer(features, hidden, dropout=dropout, 
                                               alpha=alpha, concat=True, nodes_num=nodes_num) \
                           for _ in range(nheads)])
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
        assert not torch.isnan(feat_matrix).any()
        feat_matrix = F.dropout(feat_matrix, self.dropout, training=self.training)
        feat_matrix = torch.cat([att(feat_matrix, adj, index) for index,att in enumerate(self.attentions)], dim=-1) # (B,N,hidden*heads)
        feat_matrix = F.dropout(self.normal_layer(feat_matrix), self.dropout, training=self.training)
        
        logits_sent = torch.sigmoid(self.out_att_sent(feat_matrix, adj, -2))
        logits_para = F.elu(self.out_att_para(feat_matrix, adj, -3))
        logits_Qtype = F.elu(torch.matmul(self.out_att_Qtype(feat_matrix, adj, -4)[:,0,:], self.W2)).view(-1,2)
        
        assert not torch.isnan(logits_sent).any()
        assert not torch.isnan(logits_para).any()
        assert not torch.isnan(logits_Qtype).any()
        return logits_sent, logits_para, logits_Qtype # 前2个:[B, N, num_class] 最后:[B,2]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT_HotpotQA().to(device)
    
    def gen():
        for i in range(100):
            feat = torch.randn([50,200,768]).to(device)
            adj = torch.randint(0, 2, [50, 200,200]).to(device)
            label = torch.randint(0, 2, [50,200]).to(device)
            yield (feat, adj, label)

    from apex import amp
    from apex.parallel import DistributedDataParallel
    
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend='nccl',init_method='env://')
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # model = DistributedDataParallel(model)
    device_ids = eval(f"[{os.environ['CUDA_VISIBLE_DEVICES']}]")
    model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids, output_device=0, find_unused_parameters=True)
    loss_fn = nn.CrossEntropyLoss()

    for index,i in enumerate(gen()):
        optimizer.zero_grad()
        logits_sent, logits_para, logits_Qtype = model(i[0], i[1])
        loss1 = loss_fn(logits_sent.view(-1, 2), i[2].view(-1))
        loss2 = loss_fn(logits_para.view(-1, 2), i[2].view(-1))
        loss3 = loss_fn(logits_Qtype, torch.tensor([1]*50, device='cuda'))
        loss = loss1 + loss2 + loss3
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        print(index, loss)
        optimizer.step()

    print("final loss = ", loss)

"""
git pull && CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch GNN.py
"""