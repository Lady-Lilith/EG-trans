import dgl
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from PA import ParNetAttention
from torch_geometric.nn import MessagePassing, GCNConv, TransformerConv
from MA import LiteMLA
from param import parameter_parser
torch.backends.cudnn.enabled = False
from torch import nn, dropout
import torch.nn.functional as F
class mymodel(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(mymodel, self).__init__()
        self.args = args
        self.dim_feedforward=256
        self.output_dim =128
        self.max_len = 1000
        self.n_heads = 8
        self.num_layers =1
        self.bert_modelr = BERT(self.args.fcir, self.args.hidd_dim, self.n_heads, self.dim_feedforward, self.num_layers,
                                self.output_dim, self.max_len, dropout=0.5)
        self.bert_modeld = BERT(self.args.fdis, self.args.hidd_dim, self.n_heads, self.dim_feedforward, self.num_layers,
                                self.output_dim, self.max_len, dropout=0.5)
        self.bert_modelg = BERTG(self.args.fdis, self.args.hidd_dim, self.n_heads, self.dim_feedforward, self.num_layers,
                                self.output_dim, self.max_len, dropout=0.5)
        self.gcn_cir1_f = GCNConv(self.args.fcir, self.args.fcir)
        self.gcn_dis1_f = GCNConv(self.args.fdis, self.args.fdis)
        self.pa = ParNetAttention(channel=self.args.disease_number+self.args.circRNA_number,dropout=0.2)
        self.linear = nn.Linear(self.args.circRNA_number+self.args.disease_number,self.args.disease_number)
        self.linear_cir = nn.Linear(in_features=self.args.disease_number, out_features=args.fcir)
        self.linear_dis = nn.Linear(in_features=self.args.circRNA_number, out_features=args.fdis)
        # self.linear_cir = nn.Linear(in_features=self.args.circRNA_number, out_features=args.fdis)
        # self.linear_dis = nn.Linear(in_features=self.args.disease_number, out_features=args.fdis)


    def forward(self, data,crna_di_graph):
        torch.manual_seed(1)
        # x_cir = torch.randn(self.args.circRNA_number,
        #                     self.args.fcir)  # 生成随机数张量 元素符合正态分布(对称性） 形状为(args.circRNA_number,args.fcir)
        # x_dis = torch.randn(self.args.disease_number, self.args.fdis)
        x_cir = self.linear_cir(torch.tensor(data['cd']['data_matrix']))
        x_dis = self.linear_dis(torch.tensor(data['cd']['data_matrix'].t()))
        # x_cir = self.linear_cir(torch.tensor(data['cc']['data_matrix']))
        # x_dis = self.linear_dis(torch.tensor(data['dd']['data_matrix']))
        cc_edge_index = torch.tensor(data['cc']['edges'], dtype=torch.long).clone().detach()  # 边索引
        dd_edge_index = torch.tensor(data['dd']['edges'], dtype=torch.long) .clone().detach() # 边索引
        cc_edge_attr = data['cc']['data_matrix'][data['cc']['edges'][0], data['cc']['edges'][1]]  # 边特征
        dd_edge_attr = data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]]  # 边特征
        d_num_nodes = x_dis.size(0)
        # data =TransformerConv(x_dis)
        r_num_nodes = x_cir.size(0)
        cd_num_nodes = d_num_nodes+r_num_nodes
        cd_edge_index = torch.tensor(data['cd']['edges'], dtype=torch.long).clone().detach()
        cd_edge_attr = data['cd']['data_matrix'][data['cd']['edges'][0], data['cd']['edges'][1]]
        # cir = self.bert_modelr(x_cir)
        cir = self.bert_modelr(x_cir, cc_edge_index, cc_edge_attr, r_num_nodes)
        # dis_ = self.bert_modeld(x_dis)
        dis_ = self.bert_modeld(x_dis, dd_edge_index, dd_edge_attr, d_num_nodes)
        # dis_rna = self.bert_modelg(crna_di_graph, cir, dis_)
        dis_rna = self.bert_modelg(crna_di_graph, cir, dis_, cd_edge_index, cd_edge_attr, cd_num_nodes)
        f =cir.mm(dis_.t())
        f = f.mm(dis_rna.t())
        final= self.pa(f)
        # final = self.linear(f)
        return final,cir,dis_


# 定义 PositionalEncoding 类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        a = self.pe[:, :x.size(0), :]
        a = a.squeeze(0)
        x = x+a
        return self.dropout(x)


# 定义 TransformerBlock 类
class TransformerBlock(MessagePassing):
    def __init__(self, d_model, n_heads, dim_feedforward, num_nodes,dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.att = LiteMLA(in_channels=d_model, out_channels=d_model, scales=(5,))
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.num_nodes = num_nodes

    def symmetric_normalization(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index

        # 计算节点度并进行归一化处理
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0  # 处理无穷大的情况

        # 计算归一化后的边权重
        norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, norm_weight
        # self.symm = self.symmetric_normalization(num_nodes)
    def forward(self, x,edge_index,edge_attr ,mask=None):
    # def forward(self, x ):
        edge_index, edge_attr = add_self_loops(edge_index, num_nodes=x.size(0), edge_attr=edge_attr, fill_value=1)
        edge_index, edge_weight = (self.symmetric_normalization(edge_index,edge_attr,num_nodes=x.size(0)))
    # Start propagating messages
    #     src2,_= self.self_attn(x, x, x)
        src2= self.att(x)
        src = x + self.dropout(self.norm1(src2))
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(self.norm2(src2))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().clone().detach()
        edge_attr = torch.tensor(edge_weight, dtype=torch.float).clone().detach()
        #
        # 构建稀疏张量
        edge_index_sparse = SparseTensor(row=edge_index[:, 0], col=edge_index[:, 1], value=edge_attr,
                                      sparse_sizes=(x.size(0), x.size(0)))

        # 消息传递
        out = edge_index_sparse.matmul(src)
        out = self.dropout(out)
        # out = self.dropout(src)
        return out
        # return src

class TransformerBlockg(MessagePassing):
    def __init__(self, d_model, n_heads, dim_feedforward, num_nodes, dropout=0.3):
        super(TransformerBlockg, self).__init__()
        args = parameter_parser()
        self.args = args
        # self.self_attn = nn.MultiheadAttention(args.disease_number, n_heads, dropout=dropout)
        self.att = LiteMLA(in_channels=args.disease_number, out_channels=args.disease_number, scales=(5,))
        self.linear1 = nn.Linear(args.disease_number, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, args.disease_number)
        self.norm1 = nn.LayerNorm(args.disease_number)
        self.norm2 = nn.LayerNorm(args.disease_number)
        self.dropout = nn.Dropout(dropout)
        self.num_nodes = num_nodes
            # self.symm = self.symmetric_normalization(num_nodes)

    def symmetric_normalization(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, norm_weight

    # def forward(self, x ):
    def forward(self, x, edge_index, edge_attr, mask=None):
        edge_index, edge_attr = add_self_loops(edge_index, num_nodes=x.size(0), edge_attr=edge_attr, fill_value=1)
        edge_index, edge_weight = (self.symmetric_normalization(edge_index, edge_attr, num_nodes=x.size(0)))
        # Start propagating messages
        # src2, _ = self.self_attn(x, x, x)
        src2 = self.att(x)
        src = x + self.dropout(self.norm1(src2))
        src2 = self.linear2(F.relu(self.linear1(src)))
        # src2 = F.relu(self.linear1(src))
        src = src + self.dropout(self.norm2(src2))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().clone().detach()
        edge_attr = torch.tensor(edge_weight, dtype=torch.float).clone().detach()

            # 构建稀疏张量
        edge_index_sparse = SparseTensor(row=edge_index[:, 0], col=edge_index[:, 1], value=edge_attr,
                                             sparse_sizes=(x.size(0), x.size(0)))

        # 消息传递
        out = edge_index_sparse.matmul(src)
        out = self.dropout(out)
        # out = self.dropout(src)
        return out


# 定义 BERT 类self.args.fcir, self.args.hidd_dim, self.n_heads, self.dim_feedforward, self.num_layers,
#                                 self.output_dim, self.max_len, dropout=0.5
class BERT(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_nodes, dim_feedforward, num_layers, output_dim, max_len=5000,
                 dropout=0.5):
        super(BERT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.transformer_blocks = TransformerBlock(d_model, n_heads, num_nodes,dim_feedforward)
        self.output_linear = nn.Linear(d_model, 128)
    # def forward(self, x):
    def forward(self, x,edge_index,edge_attr, num_nodes):
    # def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)  # 修正了这行代码
        # x = self.transformer_blocks(x)
        x = self.transformer_blocks(x, edge_index,edge_attr,num_nodes)
        x = self.dropout(F.relu(x))
        x = self.output_linear(x)
        return x

class BERTG(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_nodes, dim_feedforward, num_layers, output_dim, max_len=5000,
                 dropout=0.5):
        super(BERTG, self).__init__()
        args = parameter_parser()
        self.args = args
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.input_projectionr = nn.Linear(128, args.disease_number)
        self.input_projectiond = nn.Linear(128, args.disease_number)
        self.pos_encoder = PositionalEncoding(args.disease_number, max_len=2000)
        self.lin = nn.Linear(d_model,self.args.disease_number)
        self.transformer_blocks = TransformerBlockg(d_model, n_heads, num_nodes,dim_feedforward)
        self.output_linear = nn.Linear(args.disease_number, args.disease_number)
    # def forward(self, g,cis,dis):
    def forward(self, g,cis,dis,edge_index,edge_attr, num_nodes):
        dis = torch.tensor(dis)
        cis = torch.tensor(cis)
        # 创建特征字典
        feature_dict = {
            'disease': dis,
            'rna': cis,
        }
        g.ndata['h'] = feature_dict
        cis = self.input_projectionr(cis)
        dis = self.input_projectiond(dis)
        feature = torch.cat((cis, dis), dim=0)
        x = self.pos_encoder(feature)  # 修正了这行代码
        # x = self.transformer_blocks(x)
        x = self.transformer_blocks(x, edge_index, edge_attr, num_nodes)
        x = self.dropout(F.relu(x))
        x = self.output_linear(x)
        return x