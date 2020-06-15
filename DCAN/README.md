# Domain Conditioned Adaptation Network

## Paper

![](./teaser.jpg)

[Domain Conditioned Adaptation Network](https://arxiv.org/abs/2005.06717) 
(AAAI Conference on Artificial Intelligence, 2020)

If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/2005.06717):
```
@inproceedings{Li20DCAN,
    title = {Domain Conditioned Adaptation Network},
    author = {Li, Shuang and Liu, Chi Harold and Lin, Qiuxia and Xie, Binhui and Ding, Zhengming and Huang, Gao and Tang, Jian},
    booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20)},    
    year = {2020}
}
```

## Prerequisites
The code is implemented with Python(3.7) and Pytorch(1.2.0).

To install the required python packages, run

```pip install -r requirements.txt ```

## Datasets
### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).
### DomainNet
DomainNet dataset can be found [here](http://ai.bu.edu/M3SDA/).
### Office-31
Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/).


## Pre-trained models
Pre-trained models can be downloaded [here]() and put in <root_dir>/pretrained_models


## Running the code
[DomainNet](http://ai.bu.edu/M3SDA/)
```
python train_dcan.py --gpu_id id --net 50/101/152 --output_path snapshot/ --data_set domainnet --source_path /data/list/domainnet/clipart_train.txt --target_path data/list/domainnet/infograph_train.txt --test_path data/list/domainnet/infograph_test.txt
```

[Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
```
python train_dcan.py --gpu_id id --net 50 --output_path snapshot/ --data_set home --source_path data/list/home/Art_65.txt --target_path data/list/home/Clipart_65.txt --test_path data/list/home/Clipart_65.txt
```

[Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
```
python train_dcan.py --gpu_id id --net 50 --output_path snapshot/ --data_set office --source_path data/list/office/amazon_31.txt --target_path data/list/office/webcam_31.txt --test_path data/list/office/webcam_31.txt
```

[ImageCLEF](https://imageclef.org/2014/adaptation)
```
python3 train_gdcan.py --gpu_id id --net 50 --output_path snapshot/ --data_set office --source_path data/list/clef/i_12.txt --target_path data/list/clef/p_12.txt --test_path data/list/clef/p_12.txt
```

## Acknowledgements
This code is heavily borrowed from [Xlearn](https://github.com/thuml/Xlearn) and [CDAN](https://github.com/thuml/CDAN).


## Contact
If you have any problem about our code, feel free to contact
- shuangli@bit.edu.cn

or describe your problem in Issues.
