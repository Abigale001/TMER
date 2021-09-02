# Temporal Meta-path Guided Explainable Recommendation (WSDM2021)

## TMER
Code of paper "[Temporal Meta-path Guided Explainable Recommendation](https://arxiv.org/pdf/2101.01433.pdf)".


## Requirements
python==3.6.12 <br>  networkx==2.5 <br> numpy==1.15.0 <br> pandas==1.0.1 <br> pytorch==1.0.0 <br> pytorch-nlp==0.5.0
<br>gensim==3.8.3

You can also install the environment via `requirements.txt` and `environment.yaml`.


## Data Preparation
The original data can be found in the [amazon data website](https://nijianmo.github.io/amazon/index.html). 

For example, the `meta_Musical_Instruments.json` of Amazon_Music can be found [here](https://forms.gle/UEkkJs69e7Z5A5Ps9).
The `user_rate_item.csv` in the code is [here](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments.csv) (ratings only).


## Usage
If you want to change the dataset, you can modify the name in the code.

1.process data (You can ignore this step, if you just want to check TMER.)

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
or

```
@inproceedings{10.1145/3437963.3441762,
	author = {Chen, Hongxu and Li, Yicong and Sun, Xiangguo and Xu, Guandong and Yin, Hongzhi},
	title = {Temporal Meta-Path Guided Explainable Recommendation},
	year = {2021},
	booktitle = {Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
	pages = {1056–1064}
}
```

