import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from drugcell_NN import *
import argparse
import gc


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):

    term_mask_map = {}

    for term, gene_set in term_direct_gene_map.items():

        mask = torch.zeros(len(gene_set), gene_dim)

        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1

        mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

        term_mask_map[term] = mask_gpu

    return term_mask_map

# solution for 1/2||x-y||^2_2 + c||x||_0
def proximal_l0(yvec, c):
    yvec_abs =  torch.abs(yvec)
    csqrt = torch.sqrt(2*c)
    
    xvec = (yvec_abs>=csqrt)*yvec
    return xvec

# solution for 1/2||x-y||^2_2 + c||x||_g
def proximal_glasso_nonoverlap(yvec, c):
    ynorm = torch.norm(yvec, p='fro')
    if ynorm > c:
        xvec = (yvec/ynorm)*(ynorm-c)
    else:
        xvec = torch.zeros_like(yvec)
    return xvec

# solution for ||x-y||^2_2 + c||x||_2^2
def proximal_l2(yvec, c):
    return (1./(1.+c))*yvec

# prune the structure by palm
def optimize_palm(model, dG, root, reg_l0, reg_glasso, reg_decay, lr=0.001, lip=0.001):
    dG_prune = dG.copy()
    for name, param in model.named_parameters():
        if "direct" in name:
            # mutation side
            # l0 for direct edge from gene to term
            param_tmp = param.data - lip*param.grad.data
            param_tmp2 = proximal_l0(param_tmp, torch.tensor(reg_l0*lip))
            #("%s: before #0 is %d, after #0 is %d, threshold: %f" %(name, len(torch.nonzero(param.data, as_tuple =False)), len(torch.nonzero(param_tmp2, as_tuple =False)), reg_l0*lip))
            param.data = param_tmp2
        elif "GO_linear_layer" in name:
            # group lasso for
            dim = model.num_hiddens_genotype
            term_name = name.split('_')[0]
            child = model.term_neighbor_map[term_name]
            for i in range(len(child)):
                #dim = model.num_hiddens_genotype
                term_input = param.data[:,i*dim:(i+1)*dim]
                term_input_grad = param.grad.data[:,i*dim:(i+1)*dim]
                term_input_tmp = term_input - lip*term_input_grad
                term_input_update = proximal_glasso_nonoverlap(term_input_tmp, reg_glasso*lip)
                #print("%s child %d: before norm is %f, after #0 is %f, threshold %f" %(name, i, torch.norm(term_input, p='fro'), torch.norm(term_input_update, p='fro'), reg_glasso*lip))
                param.data[:,i*dim:(i+1)*dim] = term_input_update
                num_n0 =  len(torch.nonzero(term_input_update, as_tuple =False))
                if num_n0 == 0 :
                    dG_prune.remove_edge(term_name, child[i])
            # weight decay for direct
            direct_input = param.data[:,len(child)*dim:]
            direct_input_grad = param.grad.data[:,len(child)*dim:]
            direct_input_tmp = direct_input - lr*direct_input_grad
            direct_input_update = proximal_l2(direct_input_tmp, reg_decay)
            param.data[:,len(child)*dim:] = direct_input_update
        else:
            # other param weigth decay
            param_tmp = param.data - lr*param.grad.data
            param.data = proximal_l2(param_tmp, 2*reg_decay*lr)
    #sub_dG_prune = dG_prune.subgraph(nx.shortest_path(dG_prune.to_undirected(),root))
    #print("Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))
    #NodesLeft = list()
    #for nodetmp in dG_prune.nodes:
    #    for path in nx.all_simple_paths(dG_prune, source=root, target=nodetmp):
    #        #print(path)
    #        NodesLeft.extend(path)
    #NodesLeft = list(set(NodesLeft))
    #sub_dG_prune = dG_prune.subgraph(NodesLeft)
    #print("Pruned   graph has %d nodes and %d edges" % (sub_dG_prune.number_of_nodes(), sub_dG_prune.number_of_edges()))
    
    del param_tmp, param_tmp2, child, term_input, term_input_grad, term_input_tmp, term_input_update
    del direct_input, direct_input_grad, direct_input_tmp, direct_input_update
    
# check network statisics
def check_network(model, dG, root):
    dG_prune = dG.copy()
    for name, param in model.named_parameters():
        if "GO_linear_layer" in name:
            # group lasso for
            dim = model.num_hiddens_genotype
            term_name = name.split('_')[0]
            child = model.term_neighbor_map[term_name]
            for i in range(len(child)):
                #dim = model.num_hiddens_genotype
                term_input = param.data[:,i*dim:(i+1)*dim]
                num_n0 =  len(torch.nonzero(term_input, as_tuple =False))
                if num_n0 == 0 :
                    dG_prune.remove_edge(term_name, child[i])
    print("Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))
    #sub_dG_prune = dG_prune.subgraph(nx.shortest_path(dG_prune.to_undirected(),root))
    NodesLeft = list()
    for nodetmp in dG_prune.nodes:
        for path in nx.all_simple_paths(dG_prune, source=root, target=nodetmp):
            #print(path)
            NodesLeft.extend(path)
    NodesLeft = list(set(NodesLeft))
    #print(Nodes)
    sub_dG_prune = dG_prune.subgraph(NodesLeft)
    print("Pruned   graph has %d nodes and %d edges" % (sub_dG_prune.number_of_nodes(), sub_dG_prune.number_of_edges()))
    
    num_node = sub_dG_prune.number_of_nodes()
    num_edge = sub_dG_prune.number_of_edges()
    
    return sub_dG_prune, num_node, num_edge
    
def check_parameter(model, CUDA_ID):
    count = torch.tensor([0]).cuda(CUDA_ID)
    for name, param in model.named_parameters():
        if "GO_linear_layer" in name:
            print(name)
            print(param.data)
            count = count + 1
            if count >= 10:
                break

        

def training_acc(model, optimizer, train_loader, train_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID):
    #Train
    model.train()
    train_predict = torch.zeros(0,0).cuda(CUDA_ID)

    for i, (inputdata, labels) in enumerate(train_loader):
        cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer

        cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
        cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)
        
        print(i)
        # Here term_NN_out_map is a dictionary
        aux_out_map, _ = model(cuda_cell_features, cuda_drug_features)

        if train_predict.size()[0] == 0:
            train_predict = aux_out_map['final'].data
        else:
            train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

        total_loss = 0
        for name, output in aux_out_map.items():
            loss = nn.MSELoss()
            if name == 'final':
                total_loss += loss(output, cuda_labels)
            else: # change 0.2 to smaller one for big terms
                total_loss += 0.2 * loss(output, cuda_labels)
        print(i, total_loss)
        
    train_corr = spearman_corr(train_predict, train_label_gpu)
        
    print("pretrained model %f total loss, %f training acc" % (total_loss, train_corr))
        
        
def test_acc(model, test_loader, test_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID):
    model.eval()
        
    test_predict = torch.zeros(0,0).cuda(CUDA_ID)

    for i, (inputdata, labels) in enumerate(test_loader):
        # Convert torch tensor to Variable
        cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
        cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)

        cuda_cell_features.cuda(CUDA_ID)
        cuda_drug_features.cuda(CUDA_ID)

        aux_out_map, _ = model(cuda_cell_features, cuda_drug_features)

        if test_predict.size()[0] == 0:
            test_predict = aux_out_map['final'].data
        else:
            test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

    test_corr = spearman_corr(test_predict, test_label_gpu)
    del aux_out_map, inputdata, labels, test_predict, cuda_cell_features, cuda_drug_features
    torch.cuda.empty_cache()
    
    #print("pretrained model %f test acc" % (test_corr))
    return test_corr
    
def retrain(model, train_loader, train_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID, learning_rate):

    for name, param in model.named_parameters():
        if "direct" in name:
            # mutation side
            # l0 for direct edge from gene to term
            mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
            param.register_hook(lambda grad: grad.mul_(mask))
        if "GO_linear_layer" in name:
            # group lasso for
            mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
            param.register_hook(lambda grad: grad.mul_(mask))

 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    for retain_epoch in range(1):
        model.train()
        train_predict = torch.zeros(0,0).cuda(CUDA_ID)

        best_acc = [0]
        for i, (inputdata, labels) in enumerate(train_loader):
            cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer

            cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
            cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)

            # Here term_NN_out_map is a dictionary
            aux_out_map, _ = model(cuda_cell_features, cuda_drug_features)

            if train_predict.size()[0] == 0:
                train_predict = aux_out_map['final'].data
            else:
                train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

            total_loss = 0
            for name, output in aux_out_map.items():
                loss = nn.MSELoss()
                if name == 'final':
                    total_loss += loss(output, cuda_labels)
                else: # change 0.2 to smaller one for big terms
                    total_loss += 0.2 * loss(output, cuda_labels)
            optimizer.zero_grad()
            print("Retrain %d: total loss %f" % (i, total_loss.item()))
            total_loss.backward()
    
            optimizer.step()
            print("Retrain %d: total loss %f" % (i, total_loss.item()))
            
        train_corr = spearman_corr(train_predict, train_label_gpu)
        retrain_test_corr = test_acc(model, test_loader, test_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID)
        print(">>>>>Retraining step %d: model test acc %f" % (retain_epoch, prune_test_corr))
        
        if retrain_test_corr > best_acc[-1]:
            best_acc.append(accuracy)
            torch.save(model.state_dict(), model_save_folder + 'prune_final/drugcell_retrain_lung_best'+str(epoch)+'_'+str(retain_epoch)+'.pkl')
            best_model = model.state_dict()
            
        model.load_state_dict(best_model)
    return model
    
def grad_hook_masking(grad, mask):
    grad = grad.mul_(mask)
    del mask
    #return grad.mul_(mask)

def sparse_direct_gene(model, GOlist):
    GO_direct_spare_gene = {}
    for go in GOlist:
        GO_direct_spare_gene[go] = list()
    
    preserved_gene = list()
    for name, param in model.named_parameters():
        if "direct" in name:
            GOname = name.split('_')[0]
            if GOname in GOlist:
                param_tmp = torch.sum(param.data, dim=0)
                #print(param_tmp.shape)
                indn0 = torch.nonzero(param_tmp, as_tuple=True)[0]
                #param_tmp = np.sum(param.data.numpy(), axis=0)
                #print(param_tmp.shape)
                #indn0 = np.nonzero(param_tmp)[0]
                #print(np.count_nonzero(param_tmp))
                GO_direct_spare_gene[GOname].extend(indn0)
                preserved_gene.extend(indn0)
                
    return GO_direct_spare_gene, preserved_gene

# train a DrugCell model 
def train_model(pretrained_model, root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, drug_dim, model_save_folder, train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features):

    '''
    # arguments:
    # 1) root: the root of the hierarchy embedded in one side of the model
    # 2) term_size_map: dictionary mapping the name of subsystem in the hierarchy to the number of genes contained in the subsystem
    # 3) term_direct_gene_map: dictionary mapping each subsystem in the hierarchy to the set of genes directly contained in the subsystem (i.e., children subsystems would not have the genes in the set)
    # 4) dG: the hierarchy loaded as a networkx DiGraph object
    # 5) train_data: torch Tensor object containing training data (features and labels)
    # 6) gene_dim: the size of input vector for the genomic side of neural network (visible neural network) embedding cell features 
    # 7) drug_dim: the size of input vector for the fully-connected neural network embedding drug structure 
    # 8) model_save_folder: the location where the trained model will be saved
    # 9) train_epochs: the maximum number of epochs to run during the training phase
    # 10) batch_size: the number of data points that the model will see at each iteration during training phase (i.e., #training_data_points < #iterations x batch_size)
    # 11) learning_rate: learning rate of the model training
    # 12) num_hiddens_genotype: number of neurons assigned to each subsystem in the hierarchy
    # 13) num_hiddens_drugs: number of neurons assigned to the fully-connected neural network embedding drug structure - one string containing number of neurons at each layer delimited by comma(,) (i.e. for 3 layer of fully-connected neural network containing 100, 50, 20 neurons from bottom - '100,50,20')
    # 14) num_hiddens_final: number of neurons assigned to the fully-connected neural network combining the genomic side with the drug side. Same format as 13).
    # 15) cell_features: a list containing the features of each cell line in tranining data. The index should match with cell2id list.
    # 16) drug_features: a list containing the morgan fingerprint (or other embedding) of each drug in training data. The index should match with drug2id list.
    '''
    #print("Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))
    # initialization of variables
    best_model = 0
    #best_model = 0
    max_corr = 0
    dGc = dG.copy()
    
    # driver gene
    DgeneId = [51,125,128,140,171,184,214,261,281,283,287,372,378,468
,498,620,712,801,822,834,846,850,871,872,879,950,951,1082
,1131,1212,1247,1265,1305,1466,1497,1514,1516,1517,1520,1561,1607,1610
,1611,1657,1767,1790,1836,1885,1887,2016,2017,2062,2066,2113,2186,2197
,2207,2263,2289,2291,2344,2351,2357,2366,2465,2469,2612,2618,2829,2832]

    # separate the whole data into training and test data
    train_feature, train_label, test_feature, test_label = train_data

    # copy labels (observation) to GPU - will be used to
    train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
    test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))
    
    # create dataloader for training/test data
    train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
    test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)
    
    # create a torch objects containing input features for cell lines and drugs
    cuda_cells = torch.from_numpy(cell_features)
    cuda_drugs = torch.from_numpy(drug_features)
    
    # dcell neural network
    model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, CUDA_ID)
    
    # load model to GPU
    model.cuda(CUDA_ID)

    # define optimizer
    # optimize drug NN
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)
        
    # load pretrain model
    if os.path.isfile(pretrained_model):
        print("Pre-trained model exists:" + pretrained_model)
        model.load_state_dict(torch.load(pretrained_model,map_location=torch.device('cuda', CUDA_ID))) #param_file
        #base_test_acc = test(model,val_loader,device)
    else:
        print("Pre-trained model does not exist, so before pruning we have to pre-train a model.")
        sys.exit()
    #training_acc(model, optimizer, train_loader, train_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID)
    #test_acc(model, test_loader, test_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID)
    

    

    '''optimizer.zero_grad()
    for name, param in model.named_parameters():
        term_name = name.split('_')[0]

        if '_direct_gene_layer.weight' in name:
	    #print(name, param.size(), term_mask_map[term_name].size()) 
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
	    #param.data = torch.mul(param.data, term_mask_map[term_name])
        else:
            param.data = param.data * 0.1
    '''

    #training_acc(model, optimizer, train_loader, train_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID)
    #test_acc(model, test_loader, test_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID)

    #best_prune_acc = torch.tensor(0.0)
    for epoch in range(train_epochs):

        # prune step
        for prune_epoch in range(10):
	        #Train
            model.train()
            train_predict = torch.zeros(0,0).cuda(CUDA_ID)

            for i, (inputdata, labels) in enumerate(train_loader):
                cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))
                
	            # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer

                cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
                cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)
                
                cuda_cell_features.cuda(CUDA_ID)
                cuda_drug_features.cuda(CUDA_ID)

	            # Here term_NN_out_map is a dictionary
                aux_out_map, _ = model(cuda_cell_features, cuda_drug_features)

                if train_predict.size()[0] == 0:
                    train_predict = aux_out_map['final'].data
                else:
                    train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

                total_loss = 0
                for name, output in aux_out_map.items():
                    loss = nn.MSELoss()
                    if name == 'final':
                        total_loss += loss(output, cuda_labels)
                    else: # change 0.2 to smaller one for big terms
                        total_loss += 0.2 * loss(output, cuda_labels)

                total_loss.backward()

                for name, param in model.named_parameters():
                    if '_direct_gene_layer.weight' not in name:
                        continue
                    term_name = name.split('_')[0]
                    #print(name, param.grad.data.size(), term_mask_map[term_name].size())
                    param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
          
                #print("Original graph has %d nodes and %d edges" % (dGc.number_of_nodes(), dGc.number_of_edges()))
                optimize_palm(model, dGc, root, reg_l0=0.0001, reg_glasso=0.1, reg_decay=0.001, lr=0.001, lip=0.001)
                print("check network:")
                #check_network(model, dGc, root)
                #optimizer.step()
                print("Prune %d: total loss %f" % (i,total_loss.item()))
            del total_loss, cuda_cell_features, cuda_drug_features
            del aux_out_map, inputdata, labels
            torch.cuda.empty_cache()

            train_corr = spearman_corr(train_predict, train_label_gpu)
            prune_test_corr = test_acc(model, test_loader, test_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID)
            print(">>>>>%d epoch run Pruning step %d: model train acc %f test acc %f" % (epoch, prune_epoch, train_corr, prune_test_corr))
            del train_predict, prune_test_corr
            torch.cuda.empty_cache()
        

        # retraining step
        #retrain(model, train_loader, train_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID, learning_rate)
        # masking
        '''
        print("check network before masking:")
        check_network(model, dGc, root)
        handle_list = list()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "direct" in name:
                    # mutation side
                    # l0 for direct edge from gene to term
                    mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                    handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                    #handle = param.register_hook(lambda grad: grad.mul_(torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))))
                    handle_list.append(handle)
                if "GO_linear_layer" in name:
                    # group lasso for
                    mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                    handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                    #handle = param.register_hook(lambda grad: grad.mul_(torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))))
                    handle_list.append(handle)
        torch.cuda.empty_cache()
        '''
        
        #print("check network after masking:")
        #check_network(model, dGc, root)
         
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
        
        #print("check network after optimizer:")
        #check_network(model, dGc, root)
     
        best_retrain_corr = torch.tensor(0.0).cuda(CUDA_ID)
        for retain_epoch in range(10):
        
            #print("check network before train:")
            #check_network(model, dGc, root)
            
            print("check network before masking:")
            #check_network(model, dGc, root)
            # add hooks
            handle_list = list()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "direct" in name:
                        # mutation side
                        # l0 for direct edge from gene to term
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        #handle = param.register_hook(lambda grad: grad.mul_(torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))))
                        handle_list.append(handle)
                    if "GO_linear_layer" in name:
                        # group lasso for
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        #handle = param.register_hook(lambda grad: grad.mul_(torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))))
                        handle_list.append(handle)
            
            
            model.train()
            train_predict = torch.zeros(0,0).cuda(CUDA_ID)
            
            #print("check network before retrain:")
            #check_network(model, dGc, root)

            best_acc = torch.tensor([0]).cuda(CUDA_ID)
            for i, (inputdata, labels) in enumerate(train_loader):
                cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer

                cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
                cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)
                
                cuda_cell_features.cuda(CUDA_ID)
                cuda_drug_features.cuda(CUDA_ID)

                # Here term_NN_out_map is a dictionary
                aux_out_map, _ = model(cuda_cell_features, cuda_drug_features)

                if train_predict.size()[0] == 0:
                    train_predict = aux_out_map['final'].data
                else:
                    train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

                total_loss = 0
                for name, output in aux_out_map.items():
                    loss = nn.MSELoss()
                    if name == 'final':
                        total_loss += loss(output, cuda_labels)
                    else: # change 0.2 to smaller one for big terms
                        total_loss += 0.2 * loss(output, cuda_labels)
                optimizer.zero_grad()
                #print("Retrain %d: total loss %f" % (i, total_loss.item()))
                total_loss.backward()
            
                print("@check network before step:")
                #check_network(model, dGc, root)
                #check_parameter(model, CUDA_ID)
                optimizer.step()
                #check_parameter(model, CUDA_ID)
                print("@check network after step:")
                #check_network(model, dGc, root)
                print("Retrain %d: total loss %f" % (i, total_loss.item()))
                
            del total_loss, cuda_cell_features, cuda_drug_features
            del aux_out_map, inputdata, labels
            torch.cuda.empty_cache()
            
            # remove hooks
            for handle in handle_list:
                handle.remove()
            torch.cuda.empty_cache()

            gc.collect()

            train_corr = spearman_corr(train_predict, train_label_gpu)
            retrain_test_corr = test_acc(model, test_loader, test_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID)
            print(">>>>>%d epoch Retraining step %d: model training acc %f test acc %f" % (epoch, retain_epoch, train_corr, retrain_test_corr))
            
        # save models
        if best_retrain_corr < retrain_test_corr:
            best_retrain_corr = retrain_test_corr
            PrunedG, NumNode_left, NumEdge_left = check_network(model, dGc, root)
            GOLeft = list(PrunedG.nodes)
            GO_direct_gene, Prev_gene_tmp = sparse_direct_gene(model, GOLeft)
            Prev_gene = [Prev_gene_tmp[i].item() for i in range(len(Prev_gene_tmp))]
            Prev_gene_unique = list(set(Prev_gene))
            NumGeneLeft = len(Prev_gene_unique)
            Overlap = list(set(Prev_gene_unique) & set(DgeneId))
            NumOverlap = len(Overlap)
            fname =  model_save_folder + 'st_gl_0.1_epoch_'+str(epoch)+'_trainacc_'+str(train_corr.item())+'_testacc_'+str(retrain_test_corr.item())+'_nodeleft_'+str(NumNode_left)+'_edgeleft_'+str(NumEdge_left)+'_geneleft_'+str(NumGeneLeft)+'_overlap_'+str(NumOverlap)+'.pkl'
            torch.save(model.state_dict(), fname)
            
            #if retrain_test_corr > best_acc:
            #    best_acc = retrain_test_corr
                #best_acc.append(retrain_test_corr)
                #torch.save(model.state_dict(), model_save_folder + 'prune_final/drugcell_retrain_lung_best'+str(epoch)+'_'+str(retain_epoch)+'.pkl')
            #    best_model_para = model.state_dict()
                
            #model.load_state_dict(best_model_para)
            #del best_model_para
            
    

            
            
            
            

        


parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-train', help='Training dataset', type=str)
parser.add_argument('-test', help='Validation dataset', type=str)
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=3000)
parser.add_argument('-modeldir', help='Folder for trained models', type=str, default='MODEL/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)

parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=3)
parser.add_argument('-drug_hiddens', help='Mapping for the number of neurons in each layer', type=str, default='100,50,3')
parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=3)

parser.add_argument('-cellline', help='Mutation information for cell lines', type=str)
parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str)
parser.add_argument('-pretrained_model', help='Pre-trained drugcell baseline model', type=str)

print("Start....")

# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=3)

# load input data
train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(opt.train, opt.test, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)
print('Total number of genes = %d' % len(gene2id_mapping))

cell_features = np.genfromtxt(opt.cellline, delimiter=',')
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])


# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)
#print("Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens

num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))

num_hiddens_final = opt.final_hiddens

# load pretrain model
pretrained_model = opt.pretrained_model
######################################

# driver gene
#Dgene = [ 214, 1082, 1466, 1520, 1531, 1607, 2062, 2773, 2832]


CUDA_ID = opt.cuda

#print(">>>>>>>>>>>>Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))
train_model(pretrained_model, root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, drug_dim, opt.modeldir, opt.epoch, opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features)
