#!/bin/bash
codedir="/home/yijwang/Drugcell_GCN_NN/DrugCell_Prune/DrugCell_Prune/"
inputdir="/home/yijwang/Drugcell_GCN_NN/data"
ontfile="/home/yijwang/Drugcell_GCN_NN/data/go_bp_drugcell_min10_merge30_depth5_ontology.txt"
gene2idfile="/home/yijwang/Drugcell_GCN_NN/data/cell_mutations_bp_gene2id.txt"
drug2idfile="/home/yijwang/Drugcell_GCN_NN/data/drug_fingerprints_drug2id.txt"
cell2idfile="/home/yijwang/Drugcell_GCN_NN/data/cell_mutations_cell2id.txt"
celllinefile="/home/yijwang/Drugcell_GCN_NN/data/cell_mutations_bp_matrix.txt"
drugfile="/home/yijwang/Drugcell_GCN_NN/data/drug_fingerprints_matrix.txt"
modeldir="/home/yijwang/Drugcell_GCN_NN/MODEL/"

#foldid=$1
cudaid=$1

#modeldir=MODEL_$foldid
#mkdir $modeldir

python -u $codedir/train_drugcell_prune.py  -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $inputdir/drugcell_LUNG_train.txt -test $inputdir/drugcell_LUNG_test.txt -model $modeldir -cuda $cudaid -cellline $celllinefile -fingerprint $drugfile -genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 
#> train_drugcell_prune_LUNG.log
