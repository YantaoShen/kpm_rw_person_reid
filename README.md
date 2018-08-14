# Person-ReID with Deep Kronecker-Product Matching and Group-shuffiling Random Walk

This is a Pytorch implementation of our two CVPR 2018 works' combination:

* End-to-End Deep Kronecker-Product Matching for Person Re-identification (KPM) [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_End-to-End_Deep_Kronecker-Product_CVPR_2018_paper.pdf)  
* Deep Group-shuffling Random Walk for Person Re-identification (GSRW) [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Deep_Group-Shuffling_Random_CVPR_2018_paper.pdf)

Our code is mainly based on [open-reid](https://github.com/Cysu/open-reid)


## Requirements
* python 2.7 (We recommend to use [Anaconda](https://www.anaconda.com/download/#linux), since many python libs like [numpy](http://www.numpy.org/) and [sklearn](http://scikit-learn.org/stable/) are needed in our code.)  
* [PyTorch](https://pytorch.org/previous-versions/) (we run the code under version 0.3.0, maybe versions <= 0.3.1 also work.)   
* [metric-learn](https://github.com/metric-learn/metric-learn)  

Then you can clone our git repo with
```shell
git clone https://github.com/YantaoShen/kpm_rw_person_reid.git
cd kpm_rw_person_reid
python setup.py install
```

## Datasets Download
We conduct experiments on [Market1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [CUHK03](//docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0), and [DukeMTMC](https://drive.google.com/uc?id=0B0VOCNYh8HeRdnBPa2ZWaVBYSVk) datasets.

You can download these datasets as `.zip` through the links above, then   
```shell
cd examples/
mkdir data
cd data/
mdkir market1501
cd market1501
mkdir raw/
mv dir_of_market1501_zip raw/
```
For CUHK03 and DukeMTMC, the process is the same, the code will unzip these `.zip` files automatically.


## Examples
For training the model with KPM and GSRW, please download our pretrained models on three datasets, which are `baseline model + KPM` in the paper (group number is 2).  
* [Market1501_pre_model](https://drive.google.com/open?id=1NKjvf5FbLR2eqybFULBRc3f2_9KM8J4W) 
* [CUHK03_pre_model](https://drive.google.com/open?id=1cKwO7ra9QJsjja5GtgpeFNJyg-sTs6ba) 
* [DukeMTMC_pre_model](https://drive.google.com/open?id=1RhouE85aji9w7asdPolmGbWnQsZBVAIm).

Then you can train the model with follow commands
```shell
python examples/main.py -d cuhk03 -b 88 --features 2048 --alpha 0.95 --grp-num 2 --lr 0.000001 --ss 10 --epochs 10 --dropout 0.8 --combine-trainval --weight-decay 0 --retrain examples/logs/cuhk03-pretrained/model_best.pth.tar --logs-dir examples/logs/cuhk03-final-model
```

We trained this model on a server with 8 TITAN X GPUs. if you don't have such or better hardware. You may decrease the batchsize (the performance may also drop).

Or you can directly download our final model 
* [Market1501_final_model](https://drive.google.com/open?id=1yV6gX12w7SaTwF9BfyO2F1x3Ky0JjZUS)
* [CUHK03_final_model](https://drive.google.com/open?id=1Qzu7JmNkeiol0XK1u_yURE-IqG8lBRkU)
* [DukeMTMC_final_model](https://drive.google.com/open?id=1DEEZnriHpKLq8ntr_Ly3g5VpI2RnAGHH).

And test them with follow commands on different datasets
```shell
python examples/main.py -d cuhk03 -b 256 --features 2048 --alpha 0.95 --grp-num 2 --resume ./examples/logs/cuhk03-final-model/model_best.pth.tar --evaluate
```

## License and Citation
This code is released under MIT license.

Please cite these papers in your publications if it helps your research:
```
@inproceedings{shen2018deep,
  title={Deep Group-Shuffling Random Walk for Person Re-Identification},
  author={Shen, Yantao and Li, Hongsheng and Xiao, Tong and Yi, Shuai and Chen, Dapeng and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2265--2274},
  year={2018}
}
```

```
@inproceedings{shen2018end,
  title={End-to-End Deep Kronecker-Product Matching for Person Re-Identification},
  author={Shen, Yantao and Xiao, Tong and Li, Hongsheng and Yi, Shuai and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6886--6895},
  year={2018}
}
```




