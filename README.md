# ST-TGExplainer: Disentangling Stability and Transition Patterns for Temporal GNN Interpretability (ICML2026)
This repository is built for the paper [ST-TGExplainer: Disentangling Stability and Transition Patterns for Temporal GNN Interpretability](http://arxiv.org/abs/2605.19822).


### Requirements
Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Dataset and Preprocessing
#### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```data/```.

#### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```


## Model Training
```shell
python train_link_prediction.py  --dataset_name wikipedia
```

## Acknowledgments
We are grateful to the authors of [DyGLib](https://github.com/yule-BUAA/DyGLib), [TGB-Seq](https://github.com/TGB-Seq/TGB-Seq), and [TempME](https://github.com/Graph-and-Geometric-Learning/TempME).


## Citation
If you find this work useful, please consider citing:
```{bibtex}
@article{chen2026st,
  title={ST-TGExplainer: Disentangling Stability and Transition Patterns for Temporal GNN Interpretability},
  author={Chen, Hongjiang and Zheng, Xin and Jiao, Pengfei and Liu, Huan and Zhao, Zhidong and Wu, Huaming and Xia, Feng and Pan, Shirui},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```