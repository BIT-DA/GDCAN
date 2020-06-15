# DCAN
Code release for ["Domain Conditioned Adaptation Network"](https://arxiv.org/pdf/2005.06717) (AAAI 2020)

## Prerequisites
The code is implemented with Python(3.7) and Pytorch(1.2.0).

To install the required python packages, run

```pip install -r requirements.txt ```

## Training
[Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
```
python train_dcan.py --gpu_id id --net 50 --output_path snapshot/ --data_set home --source_path data/list/home/Art_65.txt --target_path data/list/home/Clipart_65.txt --test_path data/list/home/Clipart_65.txt
```

[DomainNet](http://ai.bu.edu/M3SDA/)
```
python train_dcan.py --gpu_id id --net 50/101/152 --output_path snapshot/ --data_set domainnet --source_path /data/list/domainnet/clipart_train.txt --target_path data/list/domainnet/infograph_train.txt --test_path data/list/domainnet/infograph_test.txt
```

[Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
```
python train_dcan.py --gpu_id id --net 50 --output_path snapshot/ --data_set office --source_path data/list/office/amazon_31.txt --target_path data/list/office/webcam_31.txt --test_path data/list/office/webcam_31.txt
```

## Acknowledgement
This code is implemented based on the published code of [Xlearn](https://github.com/thuml/Xlearn) and [CDAN](https://github.com/thuml/CDAN), and it is our pleasure to acknowledge their contributions.

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{Li20DCAN,
    title = {Domain Conditioned Adaptation Network},
    author = {Li, Shuang and Liu, Chi Harold and Lin, Qiuxia and Xie, Binhui and Ding, Zhengming and Huang, Gao and Tang, Jian},
    booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20)},    
    year = {2020}
}
```

## Contact
If you have any problem about our code, feel free to contact
- shuangli@bit.edu.cn

or describe your problem in Issues.
