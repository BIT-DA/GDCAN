# Generalized Domain Conditioned Adaptation Network
The GDCAN and DCAN algorithms implemented in Pytorch.
- (AAAI2020 - DCAN) [Domain Conditioned Adaptation Network](https://arxiv.org/abs/2005.06717)
- (T-PAMI - GDCAN) [Generalized Domain Conditioned Adaptation Network](https://arxiv.org/abs/2103.12339)


## Introduction
We relax a shared-convnets assumption made by previous DA methods and propose a Domain Conditioned Adaptation Network (DCAN), which aims to excite distinct convolutional channels with a domain conditioned channel attention mechanism. As a result, the critical low-level domain-dependent knowledge could be explored appropriately. Moreover, to effectively align high-level feature distributions across two domains, we further deploy domain conditioned feature correction blocks after task-specific layers, which will explicitly correct the domain discrepancy.
![](./teaser.jpg)


## Citation
If you find this work valuable or use our code in your own research, please consider citing us with the following bibtex:
```
@inproceedings{li20DCAN,
    title = {Domain Conditioned Adaptation Network},
    author = {Li, Shuang and Liu, Chi Harold and Lin, Qiuxia and Xie, Binhui and Ding, Zhengming and Huang, Gao and Tang, Jian},
    booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence},    
    pages     = {11386--11393},
    publisher = {{AAAI} Press},
    year = {2020}
}
@article{li2021generalized,
  author          = {Li, Shuang and Xie, Binhui and Lin, Qiuxia and Liu, Chi Harold and Huang, Gao and Wang, Guoren},
  title           = {Generalized Domain Conditioned Adaptation Network},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume          = {},
  number          = {},
  pages={1-1},
  year            = {2021},
  doi={10.1109/TPAMI.2021.3062644}
}
```
