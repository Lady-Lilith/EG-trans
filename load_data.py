import csv
import math
import pandas as pd
from param import parameter_parser
import dgl
import torch
import random
from train1 import train
import numpy as np
from torch import optim
args = parameter_parser()
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(cd_data)
    

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def load_dataset(args):
    dataset = dict()
    dataset['c_d'] = read_csv(args.dataset_path + '/c_d.csv')
    # dataset['c_d'] = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets2\c_d.csv")
    # dataset['c_d'] = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets\c_d.csv")

    zero_index = []
    one_index = []
    cd_pairs = []
    for i in range(dataset['c_d'].size(0)):
        for j in range(dataset['c_d'].size(1)):
            if dataset['c_d'][i][j] < 1:
                zero_index.append([i, j, 0])
            if dataset['c_d'][i][j] >= 1:
                one_index.append([i, j, 1])
   
    cd_pairs = random.sample(zero_index, len(one_index)) + one_index

    dd_matrix = read_csv(args.dataset_path + '/d_d.csv')
    # dd_matrix = read_csv(args.dataset_path + '/GDD_similarity.csv')
    # dd_matrix = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets\d_d.csv")
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}
    dis = np.array(dd_matrix)

    # cc_matrix = read_csv(args.dataset_path + '/GCC_similarity.csv')
    cc_matrix = read_csv(args.dataset_path + '/c_c.csv')
    # cc_matrix = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets2\c_c.csv")
    cc_edge_index = get_edge_index(cc_matrix)
    dataset['cc'] = {'data_matrix': cc_matrix, 'edges': cc_edge_index}
    cis = np.array(cc_matrix)

    cd_matrix = read_csv(args.dataset_path +'/c_d.csv')
    cd_edge_index = get_edge_index(cd_matrix)
    dataset['cd'] = {'data_matrix': cd_matrix, 'edges': cd_edge_index}

    return dataset, cd_pairs,cc_edge_index,dd_edge_index,one_index,cd_edge_index,dis,cis


def feature_representation(model, args, dataset):
    model
    dataset, cd_pairs, cc_edge_index, dd_edge_index, one_index, cd_edge_index, dis, cis = load_dataset(args)
    crna_di_graph = dgl_heterograph(dis,cis,one_index, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    model = train(model, dataset, optimizer, args,scheduler)
    model.eval()
    with torch.no_grad():
        score, cir_fea, dis_fea = model(dataset,crna_di_graph)
    # cir_fea = cir_fea.numpy()
    # dis_fea = dis_fea.numpy()
    return score, cir_fea, dis_fea



def new_dataset(cir_fea, dis_fea, cd_pairs):
    unknown_pairs = []
    known_pairs = []
    # [i, j, label]
    for pair in cd_pairs:
        if pair[2] == 1:
            known_pairs.append(pair[:2])

        if pair[2] == 0:
            unknown_pairs.append(pair[:2])

    print("--------------------")
    print(cir_fea.shape, dis_fea.shape)
    print("--------------------")
    # print(len(unknown_pairs), len(known_pairs))

    nega_list = []
    for i in range(len(unknown_pairs)):
        nega = cir_fea[unknown_pairs[i][0], :].tolist() + dis_fea[unknown_pairs[i][1], :].tolist() + [0, 1]
        nega_list.append(nega)

    posi_list = []
    for j in range(len(known_pairs)):
        posi = cir_fea[known_pairs[j][0], :].tolist() + dis_fea[known_pairs[j][1], :].tolist() + [1, 0]
        posi_list.append(posi)

    samples = posi_list + nega_list

    random.shuffle(samples)
    samples = np.array(samples)
    return samples


def C_Dmatix(cd_pairs,trainindex,testindex):
    args = parameter_parser()
    c_dmatix = np.zeros((args.circRNA_number,args.disease_number))
    for i in trainindex:
        if cd_pairs[i][2]==1:
            c_dmatix[cd_pairs[i][0]][cd_pairs[i][1]]=1
    
    
    dataset = dict()
    cd_data = []
    cd_data += [[float(i) for i in row] for row in c_dmatix]
    cd_data = torch.Tensor(cd_data)
    dataset['c_d'] = cd_data
    
    train_cd_pairs = []
    test_cd_pairs = []
    for m in trainindex:
        train_cd_pairs.append(cd_pairs[m])
    
    for n in testindex:
        test_cd_pairs.append(cd_pairs[n])



    return dataset['c_d'],train_cd_pairs,test_cd_pairs,cd_data
def dgl_heterograph(dis,cis,rdi, args):
    # rdi 是一个元组，包含源节点和目标节点的张量
    src_nodes = torch.tensor([item[0] for item in rdi])
    dst_nodes = torch.tensor([item[1] for item in rdi])
    # 创建节点数量字典
    node_dict = {
        'rna': args.circRNA_number,
        'disease': args.disease_number,
    }
    # 创建异构图字典
    heterograph_dict = {
        ('rna', 'association', 'disease'): (src_nodes, dst_nodes),
    }


    # 创建异构图
    crna_di_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)


    return crna_di_graph

def r_func(LD):
    # LD = LD.astype(float)
    m = LD.shape[0]
    n = LD.shape[1]
    EUC_LD = np.linalg.norm(LD, ord=2, axis=1, keepdims=False)
    EUC_DL = np.linalg.norm(LD.T, axis=1, keepdims=False)
    sum_EUC_LD = np.sum(EUC_LD)
    sum_EUC_DL = np.sum(EUC_DL)
    rl = 1/((1/m)*sum_EUC_LD)
    rt = 1/((1/n)*sum_EUC_DL)
    return rl, rt

def Gau_sim(LD, rl, rt,args):
    LD=np.mat(LD)
    DL=LD.T
    m = LD.shape[0]
    n = LD.shape[1]
    c = []
    d = []
    for i in range(m):
        for j in range(m):
            b_1 = LD[i] - LD[j]
            b_norm1 = np.linalg.norm(b_1, ord=None, axis=1, keepdims=False)
            b1 = b_norm1**2
            b1 = math.exp(-rl*b1)
            c.append(b1)
    for i in range(n):
        for j in range(n):
            b_2 = DL[i] - DL[j]
            b_norm2 = np.linalg.norm(b_2, ord=None, axis=1, keepdims=False)
            b2 = b_norm2**2
            b2 = math.exp(-rt*b2)
            d.append(b2)
    # GLL = np.mat(c).reshape(m, m)
    # GDD = np.mat(d).reshape(n, n)
    # return GLL, GDD
    GLL = np.mat(c).reshape(m, m)
    GDD = np.mat(d).reshape(n, n)

    # 将相似度矩阵保存为CSV文件
    GLL_df = pd.DataFrame(GLL)
    GDD_df = pd.DataFrame(GDD)

    GLL_df.to_csv('GCC_similarity.csv', index=False)
    GDD_df.to_csv('GDD_similarity.csv', index=False)
    return GLL, GDD
def integrate_similarities(similarity, gaussian_similarity):
    integrated_similarity = np.where(similarity == 0, gaussian_similarity,similarity)
    return integrated_similarity
# LD= pd.read_csv('data/ours/dataset2/intersection/di_lnc_intersection.csv', index_col='Unnamed: 0')
if __name__ == "__main__":
    CD = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets\c_d.csv")
    rl, rt = r_func(CD)
    GaC, GaD = Gau_sim(CD, rl, rt,args)
    gaussian_similarityC = GaC
    gaussian_similarityD = GaD
    # gaussian_similarityD = read_csv(r"D:\论文代码\论文代码\CDAModel\datasetsC\disease GIP similarity.csv" )
    # gaussian_similarityC = read_csv(r"D:\论文代码\论文代码\CDAModel\datasetsC\circRNA GIP similarity.csv")
    # gaussian_similarityC = read_csv(r"D:\论文代码\论文代码\CDAModel\dataset\Dataset-2\dissimilarity.csv")
    # disease_semantic_similarity =read_csv(r"D:\论文代码\论文代码\CDAModel\datasets3\cs.csv")
    # circrna_fun_similarity = read_csv(r"D:\论文代码\论文代码\CDAModel\datasets3\ds.csv")
    # # 定义文件路径
    # txt_file_path = r'D:\论文代码\论文代码\CDAModel\datasetsC\Association matrix.txt'
    # csv_file_path = r'D:\论文代码\论文代码\CDAModel\datasetsC\Association matrix.csv'
    #
    # # 读取TXT文件，假设是空格分隔
    # data = []
    # with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
    #     for line in txt_file:
    #         data.append(line.strip().split(','))  # 根据实际情况调整分隔符
    #
    # # 将数据转换为DataFrame
    # df = pd.DataFrame(data)
    #
    # # 保存为CSV文件
    # df.to_csv(csv_file_path, index=False)
    #
    # print(f"TXT file has been successfully converted to CSV file at {csv_file_path}")
    # exit()
    #整合相似度
    # integrated_similarityD = integrate_similarities(disease_semantic_similarity, gaussian_similarityD)
    # integrated_similarityC = integrate_similarities(circrna_fun_similarity, gaussian_similarityC)
    # integrated_similarity_dfD = pd.DataFrame(integrated_similarityD)
    # integrated_similarity_dfC = pd.DataFrame(integrated_similarityC)
    # integrated_similarity_dfD.to_csv(r"D:\论文代码\论文代码\CDAModel\newdata\d_d.csv")
    # integrated_similarity_dfC.to_csv(r"D:\论文代码\论文代码\CDAModel\newdata\c_c.csv")
# integrated_similarity_dfC.to_csv('c_c.csv', index=False)
