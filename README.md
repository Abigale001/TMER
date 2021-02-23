#Temporal Meta-path Guided Explainable Recommendation (WSDM2021)

##TMER
Code of paper "[Temporal Meta-path Guided Explainable Recommendation](https://arxiv.org/pdf/2101.01433.pdf)".


##Requirements
python==3.6.12  networkx==2.5  numpy==1.15.0  pandas==1.0.1  pytorch==1.0.0  pytorch-nlp==0.5.0

You can also install the environment via `requirements.txt` and `environment.yaml`.
##Usage
1.learn the user-item path representations

`python data/user_item_representation.py`

2.run the recommendation

`python run.py`

##Cite
If you find this code useful in your research, please consider citing:
```
@article{chen2021temporal,
  title={Temporal Meta-path Guided Explainable Recommendation},
  author={Chen, Hongxu and Li, Yicong and Sun, Xiangguo and Xu, Guandong and Yin, Hongzhi},
  journal={arXiv preprint arXiv:2101.01433},
  year={2021}
}
```

