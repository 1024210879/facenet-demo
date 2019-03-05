# facenet-demo
更改facenet源码目录的一些结构，下载即用。功能包括：一、摄像头人脸检测，关键点检测。二、静态图人脸检测，关键点检测。三、人脸对齐。四、人脸比较。五、人脸聚类。

## 运行环境

win10，python3.6.3

所需要的库有：

tensorflow、scipy、scikit-learn、opencv-python、h5py、matplotlib、Pillow、requests、psutil

## 数据准备：

① lfw数据集bd云：链接：https://pan.baidu.com/s/1mnr8WCCVCb5wrwFaH_q2UA 提取码：vgs6 
  存放路径：facenet/data/lfw

② 预训练模型bd云：https://pan.baidu.com/s/1Ejp-n_h5wC9hvB7aDojf5A 提取码：l76t
  存放路径：facenet/weight

③ 预训练.pkl文件bd云：https://pan.baidu.com/s/1rZNGK36sKuAA9dtuhpHIAg 提取码：ic5o
  存放路径：facenet/weight

## demo使用

运行bat文件

#### 摄像头人脸检测、关键点检测

运行bat_detect_face_camera.bat

#### 静态图人脸检测、关键点检测

运行bat_detect_face_images.bat

#### 人脸对齐

运行bat_align_dataset_mtcnn.bat，数据集在facenet/data/lfw，图片有点多运行完估计需要20分钟，对齐数据在facenet/data/lfw_160

#### 人脸比较

运行bat_compare.bat

#### 人脸聚类

运行bat_classifier_mode_classify.bat，没有pkl文件需要先运行bat_classifier_mode_train.bat生成pkl文件
