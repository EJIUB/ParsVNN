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


def predict_dcell(root, term_size_map, term_direct_gene_map, dG, predict_data, gene_dim, drug_dim, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, model_file, hidden_folder, result_file, cell_features, drug_features, CUDA_ID):

	device = torch.device("cuda:%d" % CUDA_ID)

	model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, CUDA_ID)
	model.cuda(CUDA_ID)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
	checkpoint = torch.load(model_file, map_location=device)

	model.load_state_dict(checkpoint['model'])
	optimizer.load_state_dict(checkpoint['optimizer'])

	model.eval()

	predict_feature, predict_label = predict_data
	predict_label_gpu = predict_label.cuda(CUDA_ID)

	cuda_cells = torch.from_numpy(cell_features)
	cuda_drugs = torch.from_numpy(drug_features)

	test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)

	#Test
	test_predict = torch.zeros(0,0).cuda(CUDA_ID)
	term_hidden_map = {}	

	batch_num = 0
	for i, (inputdata, labels) in enumerate(test_loader):
		# Convert torch tensor to Variable
		cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
		cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)


		# make prediction for test data
		aux_out_map, term_hidden_map = model(cuda_cell_features, cuda_drug_features)

		if test_predict.size()[0] == 0:
			test_predict = aux_out_map['final'].data
		else:
			test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		''' old version
		for term, hidden_map in term_hidden_map.items():
			this_hidden_file = hidden_folder+'/'+term+'_'+str(i)+'.txt'
			hidden_file = hidden_folder+'/'+term+'.hidden'

			np.savetxt(this_hidden_file, hidden_map.data.cpu().numpy(), '%.4e')	

			# append the file to the main file
			os.system(this_hidden_file + ' >> ' + hidden_file)
			os.system(this_hidden_file)
		'''

		for term, hidden_map in term_hidden_map.items():
			hidden_file = hidden_folder+'/'+term+'.hidden'
			with open(hidden_file, 'ab') as f:
				np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

		batch_num += 1

	test_corr = spearman_corr(test_predict, predict_label_gpu)
	#print 'Test pearson corr', model.root, test_corr	
	print("Test pearson corr\t%s\t%.6f" % (model.root, test_corr))

	np.savetxt(result_file+'/'+model.root+'.predict', test_predict.cpu().numpy(),'%.4e')




parser = argparse.ArgumentParser(description='DCell prediction')
parser.add_argument('-predict', help='Dataset to be predicted', type=str)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default=1000)
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default=1000)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default=1000)
parser.add_argument('-load', help='Model file', type=str, default='MODEL/model_200')
parser.add_argument('-hidden', help='Hidden output folder', type=str, default='Hidden/')
parser.add_argument('-result', help='Result file name', type=str, default='Result/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-cellline', help='Mutation information for cell lines', type=str)
parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str)

parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=3)
parser.add_argument('-drug_hiddens', help='Mapping for the number of neurons in each layer', type=str, default='100,50,3')
parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=3)



opt = parser.parse_args()
torch.set_printoptions(precision=5)

predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(opt.predict, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell features
cell_features = np.genfromtxt(opt.cellline, delimiter=',')

# load drug features
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])

CUDA_ID = opt.cuda

drug_dim = len(drug_features[0,:])
num_genes = len(gene2id_mapping)

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens

num_hiddens_drug = map(int, opt.drug_hiddens.split(','))

num_hiddens_final = opt.final_hiddens
#####################################


print("Total number of genes = %d" % len(gene2id_mapping))

predict_dcell(root, term_size_map, term_direct_gene_map, dG, predict_data, gene_dim, drug_dim, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, model_file, hidden_folder, result_file, cell_features, drug_features, CUDA_ID)


