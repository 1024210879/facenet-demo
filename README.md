# facenet-demo

## 数据准备：

①lfw数据集（lfw），bdyun下载：

链接：https://pan.baidu.com/s/1mnr8WCCVCb5wrwFaH_q2UA 

提取码：vgs6 

存放于 facenet/data

![image text]
(https://github.com/1024210879/facenet-demo/tree/master/data/images/path_lfw.jpg)

②facenet预训练权重（20170512-110547.pb），bdyun下载：

链接：https://pan.baidu.com/s/1Ejp-n_h5wC9hvB7aDojf5A 

提取码：l76t 

存放于 facenet/weight

![image text]
(https://github.com/1024210879/facenet-demo/tree/master/data/images/path_pb.jpg)

③facenet预训练pkl文件（classifier.pkl）bdyun下载：

链接：https://pan.baidu.com/s/1rZNGK36sKuAA9dtuhpHIAg 

提取码：ic5o 

存放于 facenet/weight

![image text]
(https://github.com/1024210879/facenet-demo/tree/master/data/images/path_pkl.jpg)

## 运行环境

win10，python3.6.3

运行 bat_init_env.bat 下载依赖库

## demo使用

#### 运行 bat_align_dataset_mtcnn.bat 人脸对齐 生成 data/lfw_160

#### 运行 bat_detect_face_images.bat 静态图人脸检测、人脸关键点检测

#### 运行 bat_detect_face_camera.bat 摄像头人脸检测、人脸关键点检测

#### 运行 bat_compare.bat 人脸对比

#### 运行 bat_cluster.bat 人脸聚类

#### 运行 bat_predict.bat 在人脸库查找某个人脸

#### 运行 bat_train_tripletloss.bat 训练网络
