# Augmented-Geometric-Distillation

Reproduced by GuoQuanhao using the PaddlePaddle

[paper link](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Augmented_Geometric_Distillation_for_Data-Free_Incremental_Person_ReID_CVPR_2022_paper.pdf)

[Official codel](https://github.com/eddielyc/Augmented-Geometric-Distillation)

[Aistudio repo of Paddle](https://aistudio.baidu.com/aistudio/projectdetail/5010353?sUid=91289&shared=1&ts=1669543641052)

ResNet预训练模型和训练日志模型，生成的Dreaming data已上传至[百度云盘](https://pan.baidu.com/s/1jssC6c-OEJ4ZPBnJhoahWg)，密码：**mit0**

## 数据集构建
用到的`Market-1501-v15.09.15`和`MSMT17`数据集已挂载在AIStudio和本项目
```
./data
- market
  - bounding_box_test
  - bounding_box_train
  - query
 
- msmt17
  - bounding_box_test
  - bounding_box_train
  - query
```

```
-backbone_weights
-data
-deep_inversion
-reid
-configs
-main.py
...
```

## 训练
 - 训练基于MSMT17的基础模型
```
python main.py -g 0 --dataset msmt17 --logs-dir ./logs/msmt17
```

通过DeepInversion生成dreaming data
```
python inversion.py -g 0 --generation-dir ./data/generations_r50_msmt17 --shots 40 --iters 640 --teacher ./logs/msmt17
```

采用Geometric Distillation loss训练在Market上的增量模型
```
python main_incremental.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_GD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --algo-config ./configs/res-triangle.yaml
```

采用simple Geometric Distillation loss训练在Market上的增量模型
```
python main_incremental.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_simGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --algo-config ./configs/sim-res-triangle.yaml
```

采用Augmented Distillation训练在Market上的增量模型
```
python main_incrementalX.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_XsimGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --peers 2 --algo-config ./configs/inverXion.yaml
```

## 评估
通过指定`--ckpt`评估不同的模型
```
python evaluate.py --dataset msmt17 market --ckpt ./logs/msmt17-market_XsimGD/checkpoint.pdparams --output
```

## 实现精度

|  | MSMT17-mAP | MSMT17-Rank-1 |Market-mAP | Market-Rank-1|
|:-|:-:|:-:|:-:|:-:|
| Paper | **41.9** | **67.5** | **80.5** | **91.9** |
| Geometric Distillation(reproduce) | 40.8 | 66.5 | 80.1/91.8(re-ranking) | 92.0/94.4(re-ranking)|
| simple Geometric Distillation(reproduce) | 40.6 | 65.9 | 80.1/91.8(re-ranking) | 92.3/93.8(re-ranking) |
| Augmented Distillation(reproduce) | **41.9** | **67.7** | **81.1/91.9(re-ranking)** | **92.4/94.2(re-ranking)**|

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>

| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| CSDN主页        | [Deep Hao的CSDN主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
| GitHub主页        | [Deep Hao的GitHub主页](https://github.com/GuoQuanhao) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
