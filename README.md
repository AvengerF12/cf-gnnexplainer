# CF-GNNExplainer++: Extension on Counterfactual Explanations for Graph Neural Networks

This repository is a wip implementation of an extension to the paper CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks. 


## Requirements

To install requirements:

```setup
conda env create --file environment.yml
```

>📋 This will create a conda environment called cf-gnnexplainer


## Training CF-GNNExplainer

To train CF-GNNExplainer for each dataset and obtain the results from the original paper, run the following commands:

```train
python main_explain.py --dataset=syn1 --lr=0.1 --beta=0 --n_momentum=0.9 --optimizer=SGD
python main_explain.py --dataset=syn4 --lr=0.1 --beta=0 --optimizer=SGD
python main_explain.py --dataset=syn5 --lr=0.1 --beta=0 --optimizer=SGD
```

>📋  This will create another folder in the main directory called 'results', where the results files will be stored.


## Evaluation

To evaluate the CF examples, loop over all the files in the results folder by running the following command:

```eval
python evaluate.py
```
>📋  This will print out the values for each metric, for each results file. To evaluate results from a different folder use the --res_path command line arg.


## Visualization

To visualize the CF examples run the following command:

```eval
python visualize.py --path=/path/to/cf --idx_cf="id of cf"
```
>📋  This will show the original adjacency graph, showing in black the edges that stayed the same, in green the ones that were added and in red the ones that were removed during the generation of the counterfactual.


## Pre-trained Models

The pretrained models are available in the models folder


## Results

WIP
