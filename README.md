# Predicting-circRNA-microRNA-Interactions-via-Knowledge-Graph-based-Large-Language-Model
===========================================================================


[![license](https://img.shields.io/badge/python_-3.8.0_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-1.12.0_-blue)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scanpy_-1.9.0_-blue)](https://scanpy.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/anndata_-0.8.0_-blue)](https://anndata-tutorials.readthedocs.io/en/latest/index.html/)

The interactions between circRNAs and microRNAs are one of the key mechanisms determining the functions in biological processes such as DNA methylation and RNA-induced silencing. Studying these relationships can deepen our understanding of the function of non-coding RNAs' roles in developing cancer vaccines and design treatments. Therefore,we proposed a computational method for predicting circRNA-miRNA interacrtions based on the knowledge graph enhanced pretrained large language model (LLM). The pretrained sequential LLM was trained by sequences of circRNAs and microRNAs separately to generate the semantic representation for the entities of circRNAs and microRNAs, then the graph contrastive learning was employed to generate the embedding which can represent the relationships among the knowledge graph formed by circRNAs and microRNAs from multi-view. At the final, the embedding was fed into classifier for prediction. An independent test were set to evaluate the model's performance as the remaining dataset was performed cross-validation test method's stability. The benchmark was against our model with recently reported models on two datasets. Our model achieved approximately a 3\% improvement in Area Under the Receiver Operating Characteristic Curve (AUROC), reaching 93.77\% and 93.07\%, respectively. In ablation study, we comprehensively compared  strategies for sequence processing and effectiveness of independent module. Finally, we deployed our model on an augmented dataset for a case study. Among the top ten predicted results, seven validated outcomes which can demonstrate the effectiveness of our approach in practical scenarios.


![Image text](https://github.com/jiboyalab/scDecipher/blob/main/IMG/流程图_v2.svg)


The overview of scDecipher. **(a)** The schematic diagram of a full picture of cell-cell communication (CCC). LR: Ligand-Receptor; $E$: gene expression profile; $E_{0}$: baseline expression determined by cell type; $\Delta E$: expression change caused by CCC. $F$: functional states of malignant cells; $F_{0}$: baseline state determined by the cell itself; $\Delta F$: state change caused by CCC. **(b)** The construction of multi-view CCC network at single-cell resolution. First, four excellent CCC tools were applied to infer ligand-receptor pairs from single cell expression profiles. Second, an edge weighting strategy was applied to calculate the CCC strength at single-cell resolution based on the inferred LR pairs. Third, low specificity CCCs caused by widely expressed ligands (or receptors) or sequencing technology errors were filtered out. Fourth, the CCC graphs between each pair of different cell types were constructed, where the nodes in the graph represent single cells and the edges represent the communication strength between single cells calculated in the previous step. **(c)** The multi-view graph attention learning for predicting target gene expression or functional states of malignant cells. First, each CCC graph obtained from step b was trained by a graph convolutional network (GCN) and subjected to a nonlinear transformation by ReLU function, where the input contained the CCC strength matrix as the adjacency matrix and the one-hot encoding matrix obtained according to cell types as the initial feature matrix of the nodes. Second, the gene expressions or functional states obtained under different views were fused by an attention mechanism and then retrained using a multi-layer perceptron (MLP) to estimate $\Delta E/\Delta F$. Third, another MLP was applied to estimate  $E_{0}/F_{0}$ based on the cell-type matrix. Fourth, two estimations were summed as the final prediction results and trained iteratively to minimise the mean square error (MSE) with the true gene expression profiles or functional states. **(d)** The main function of scDecipher.
## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contacts](#contacts)
- [License](#license)


## Installation

scDecipher is tested to work under:

```
* Python 3.8.0
* Torch 1.12.0
* Scanpy 1.9.0
* Anndata 0.8.0
* R 4.2.2
* Numpy 1.23.5
* Other basic python and r toolkits
```
### Installation of other dependencies
* Install [CellPhoneDB v3](https://github.com/ventolab/CellphoneDB) using ` pip install cellphonedb ` if you encounter any issue. 
* Install [CellChat v1.6.0](https://github.com/sqjin/CellChat/tree/master) using ` devtools::install_github("sqjin/CellChat") ` in the R environment if you encounter any issue.
* Install [NicheNet v1.1.0](https://github.com/saeyslab/nichenetr) using ` devtools::install_github("saeyslab/nichenetr") ` in the R environment if you encounter any issue.
* Install [ICELLNET](https://github.com/soumelis-lab/ICELLNET) using ` install_github("soumelis-lab/ICELLNET",ref="master", subdir="icellnet") ` in the R environment if you encounter any issue.


# Quick start
To reproduce our results:

## 1，infer ligand–receptor (L-R) pairs from single-cell RNA sequencing data
```
cellphonedb method statistical_analysis ./data/RCC_scRNA_P76_metadata.txt ./data/RCC_scRNA_P76_matrix.txt --counts-data=gene_name --threads 100 --output-path ./output/
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **counts-data** | [ensembl or gene_name or hgnc_symbol] |
| **threads** | Max of threads to process the data. |
| **output-path** | Directory where the results will be allocated (the directory must exist). |

```
Rscript ./tools/run_cellchat.R --count ./data/RCC_scRNA_P76_matrix.txt --meta ./data/RCC_scRNA_P76_metadata.txt  --output ./output/

# The used ligand-target matrix, lr network and weighted networks of interacting cells can be downloaded from [Zenodo](https://zenodo.org/record/7074291).
Rscript ./tools/run_nichenet.R --count ./data/RCC_scRNA_P76_matrix.txt --meta ./data/RCC_scRNA_P76_metadata.txt  --output ./output/

Rscript ./tools/run_icellnet.R --count ./data/RCC_scRNA_P76_matrix.txt --meta ./data/RCC_scRNA_P76_metadata.txt  --output ./output/
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **output** | Directory where the results will be allocated. |

```
# Obtain the intersection of LR pairs output by 4 cellular communication tools, which are required to be found by at least 2 tools and have expression in scRNA-seq data.
python ./tools/process_final_lr.py --lr_cellphonedb ./output/process_cellphonedb_lr.csv --lr_cellchat ./output/process_cellchat_lr.csv --lr_nichenet ./output/process_nichenet_lr.csv --lr_icellnet ./output/process_icellchat_lr.csv --count ./data/RCC_scRNA_P76_matrix.txt --output ./output/final_lr.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **lr_cellphonedb** | The results of LR pairs output by cellphonedb. |
| **lr_cellchat** | The results of LR pairs output by cellchat. |
| **lr_nichenet** | The results of LR pairs output by nichenet. |
| **lr_icellnet** | The results of LR pairs output by icellnet. |
| **count** | Count matrix / normalized count matrix path. |
| **output** | The final results of LR pairs. |

## 2，prioritize the dominant cell communication assmebly that regulates the target gene expression pattern
```
python ./src/tutorials1/main.py --count ./data/RCC_scRNA_P76_matrix.txt --meta ./data/RCC_scRNA_P76_metadata.txt --lr_file ./output/final_lr.csv --gene CD8A --dca_rank_result ./output/CD8A_dca_rank_result.csv --ccc_ratio_result ./output/CD8A_ccc_ratio_result.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **lr_file** | The final results of LR pairs. |
| **gene** | The specific target gene name  |
| **dca_rank_result** | The result of prioritize the dominant cell communication assmebly that regulates the target gene expression pattern. |
| **ccc_ratio_result** | The result of ratio of different cell types affected by cellular communication. |

Visualization of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDecipher/blob/main/IMG/cd8arank.png" alt="Editor" width="500">
</div>

===========================================================================

<div align="center">
  <img src="https://github.com/jiboyalab/scDecipher/blob/main/IMG/cd8adeltae.png" alt="Editor" width="400">
</div>

## 3，prioritize the dominant cell communication assmebly that regulates the key factors in specific cell type
```
python ./src/tutorials2/main.py --count ./data/RCC_scRNA_P76_matrix.txt --meta ./data/RCC_scRNA_P76_metadata.txt --lr_file ./output/final_lr.csv --gene FOLR2 --cell_type TAM --dca_rank_result ./output/FOLR2_dca_rank_result.csv --ccc_ratio_result ./output/FOLR2_ccc_ratio_result.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **lr_file** | The final results of LR pairs. |
| **gene** | The specific target gene name.  |
| **cell_type** | The specific cell type (TAM:tumor-associated macrophages).  |
| **dca_rank_result** | The result of prioritize the dominant cell communication assmebly that regulates the target gene expression pattern. |
| **ccc_ratio_result** | The result of ratio of different cell types affected by cellular communication. |

Visualization of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDecipher/blob/main/IMG/folr2tam.png" alt="Editor" width="500">
</div>

## 4，prioritize the dominant cell communication assmebly that affected functional states of malignant cells
```
python ./src/tutorials3/main.py --count ./data/RCC_scRNA_P76_matrix.txt --meta ./data/RCC_scRNA_P76_metadata.txt --lr_file ./output/final_lr.csv --cell_type Malignant --cell_state EMT --dca_rank_result ./output/state_dca_rank_result.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **lr_file** | The final results of LR pairs. |
| **cell_type** | The specific cell type.  |
| **cell_state** | [Angiogenesis; Apoptosis; CellCycle; Differentiation; DNAdamage; DNArepair; EMT; Hypoxia; Inflammation; Invasion; Metastasis; Proliferation; Quiescence; Stemness.]  |
| **dca_rank_result** | The result of prioritize the dominant cell communication assmebly that affected functional states of malignant cells. |


Visualization of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDecipher/blob/main/IMG/cellstate.png" alt="Editor" width="500">
</div>

## 5，clinical intervertion altered effect of cell communication on gene expression
```
python ./src/tutorials1/main.py --count ./data/RCC_scRNA_P915_matrix.txt --meta ./data/RCC_scRNA_P915_metadata.txt --lr_file ./output/final_lr.csv --gene CD8A --dca_rank_result ./output/P915_CD8A_dca_rank_result.csv --ccc_ratio_result ./output/P915_CD8A_ccc_ratio_result.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **lr_file** | The final results of LR pairs. |
| **gene** | The specific target gene name  |
| **dca_rank_result** | The result of prioritize the dominant cell communication assmebly that regulates the target gene expression pattern. |
| **ccc_ratio_result** | The result of ratio of different cell types affected by cellular communication. |

Visualization of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDecipher/blob/main/IMG/cd8arankchange.png" alt="Editor" width="500">
</div>

===========================================================================





# Contributing

Jiboya Xuliwen ..

# Cite
<p align="center">
  <a href="https://clustrmaps.com/site/1bpq2">
     <img width="200"  src="https://clustrmaps.com/map_v2.png?cl=ffffff&w=268&t=m&d=4hIDPHzBcvyZcFn8iDMpEM-PyYTzzqGtngzRP7_HkNs" />
   </a>
</p>

<p align="center">
  <a href="#">
     <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fjiboyalab%2FscDecipher&labelColor=%233499cc&countColor=%2370c168" />
   </a>
</p>


# Contacts
If you have any questions or comments, please feel free to email: byj@hnu.edu.cn.

# License

[MIT ? Richard McRichface.](../LICENSE)