## Low overlap point cloud registration algorithm based on coupled iteration 



### Installation
We tested the code on python 3.8.10; Pytroch version '1.7.1' or '1.9.0+cu111'; GPU model Nvidia A40.
```shell
conda env create -f environment.yml
conda activate BC-PCNet
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```


### Download data and pretrained model
- 3DMatch/3DLoMatch benchmark (from the [Predator](https://github.com/overlappredator/OverlapPredator) paper):
 [train/val/test split (4.8GB)](https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip).

- Pretrained model: .



### Train and evaluation on 3DMatch
Download and extract the 3DMatch split to your custom folder. Then update the ```data_root``` in [configs/train/3dmatch.yaml](configs/train/3dmatch.yaml) and [configs/test/3dmatch.yaml](configs/test/3dmatch.yaml)

- Evaluate pre-trained
```shell
python main.py configs/test/3dmatch.yaml
```
(To switch between 3DMatch and 3DLoMatch benchmark, modify the ```split``` configuration in  [configs/test/3dmatch.yaml](configs/test/3dmatch.yaml))


- Train from scratch
```shell
python main.py configs/train/3dmatch.yaml
```


