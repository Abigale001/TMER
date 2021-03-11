# Temporal Meta-path Guided Explainable Recommendation (WSDM2021)

## TMER
Code of paper "[Temporal Meta-path Guided Explainable Recommendation](https://arxiv.org/pdf/2101.01433.pdf)".


## Requirements
python==3.6.12 <br>  networkx==2.5 <br> numpy==1.15.0 <br> pandas==1.0.1 <br> pytorch==1.0.0 <br> pytorch-nlp==0.5.0
<br>gensim==3.8.3

You can also install the environment via `requirements.txt` and `environment.yaml`.
## Usage
If you want to change the dataset, you can modify the name in the code.

1.process data

`python data_process.py`

2.learn the user and item representations

`python data/path/embed_nodes.py`

3.learn the item-item path representations

`python data/path/user_history/item_item_representation.py`

4.learn the user-item path representations

`python data/user_item_representation.py`

5.generate user-item and item-item meta-path instances and learn their representations

`python data/path/generate_paths.py`<br>
`python data/path/user_history/meta_path_instances_representation.py`

6.sequence item-item paths for each user

`python data/path/user_history/user_history.py`

7.run the recommendation

`python run.py`


## Cite
If you find this code useful in your research, please consider citing:
```
@article{chen2021temporal,
  title={Temporal Meta-path Guided Explainable Recommendation},
  author={Chen, Hongxu and Li, Yicong and Sun, Xiangguo and Xu, Guandong and Yin, Hongzhi},
  journal={arXiv preprint arXiv:2101.01433},
  year={2021}
}
```

