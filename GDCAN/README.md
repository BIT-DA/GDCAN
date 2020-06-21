# Generalized Domain Conditioned Adaptation Network
Code release for Manuscript "Generalized Domain Conditioned Adaptation Network" (TPAMI-2020-06-0743)

## Prerequisites
The code is implemented with Python(3.7) and Pytorch(1.2.0).

To install the required python packages, run

```pip install -r requirements.txt ```

## Datasets
### DomainNet
DomainNet dataset can be found [here](http://ai.bu.edu/M3SDA/).

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### Office-31
Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/).

### ImageCLEF
ImageCLEF dataset can be found [here](https://imageclef.org/2014/adaptation)

## Pre-trained models
Pre-trained models can be downloaded [here](https://github.com/BIT-DA/GDCAN/releases) and put in <root_dir>/pretrained_models


## Running the code
[DomainNet](http://ai.bu.edu/M3SDA/)
```
python3 train_dcan.py --gpu_id id --net 50/101 --output_path snapshot/ --dset domainnet --source_path data/list/domainnet/clipart_train.txt --target_path data/list/domainnet/infograph_train.txt --test_path data/list/domainnet/infograph_test.txt --task ci 
```

[Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
```
python3 train_dcan.py --gpu_id id --net 50 --output_path snapshot/ --data_set home --source_path data/list/home/Art_65.txt --target_path data/list/home/Clipart_65.txt --test_path data/list/home/Clipart_65.txt --task ac
```

[Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
```
python3 train_dcan.py --gpu_id id --net 50 --output_path snapshot/ --data_set office --source_path data/list/office/dslr_31.txt --target_path data/list/office/webcam_31.txt --test_path data/list/office/webcam_31.txt --task dw
```

[ImageCLEF](https://imageclef.org/2014/adaptation)
```
python3 train_gdcan.py --gpu_id id --net 50 --output_path snapshot/ --data_set clef --source_path data/list/clef/c_12.txt --target_path data/list/clef/p_12.txt --test_path data/list/clef/p_12.txt --task cp --lr 5e-6 --random_prob 2.0 
```

## Contact
If you have any problem about our code, feel free to contact
- shuangli@bit.edu.cn

or describe your problem in Issues.
