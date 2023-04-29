# VQ-CodeBook

1. train a Codebook and generate the semantic map

python3 -u main.py --dataset=custom --model=vqvae --data-dir=xxx/xxx/VQ-Codebook/images

1) Semantic Map can be saved at ./Smap_train and ./Smap_test
2) The related model is saved at ./checkpoint



# SMNet

1. train a SMNet with Semantic Map

bash run_train.sh

in train.py file,

--sidd_path represents the path of train data
    clean data: xxxx/xxxx/train/GT
    noisy data: xxxx/xxxx/train/Noisy
    semantic map: xxxx/xxxx/train/Smap

--test_path represents the path of test data
    clean data: xxxx/xxxx/test/GT
    noisy data: xxxx/xxxx/test/Noisy
    semantic map: xxxx/xxxx/test/Smap

Notes: You need move the generated semantic smap file (.npy) from ./Smap_train and ./Smap_test into xxxx/xxxx/train/Smap and xxxx/xxxx/test/Smap, respectively

--save_folder represents the path of model checkpoint

Notes：The training log can be saved in log.txt.

2. test and visualization

bash run_test.sh

The visualization results are saved in ./visual_results