PU-learning-for-cell-detection
=======
Introduction
-----
The purpose of this project is to solve the problem of incomplete annotations in cell detection, implemented on the PyTorch framework.
# Preparation
## prerequisites
>* Python 2.7 or 3.6
>* Pytorch 0.4.0 (now it does not support 0.4.1 or higher)
>* CUDA 8.0 or higher
## Pretrained Model
We used two pretrained models in our experiments, VGG and ResNet. You can download these two models from:
>* VGG16:(https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)
>* ResNet101:(https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)<br>
Download them and put them into the data/pretrained_model/.
If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.
## Install all the python dependencies using pip:
```
pip install -r requirements.txt
```
## Compile the cuda dependencies using following simple commands:
```
cd lib
sh make.sh
```
# Train
Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda
```
Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. On Titan Xp with 12G memory, it can be up to 4.
# Test
If you want to evaluate the detection performance of a pre-trained model on pascal_voc test set, simply run
```
python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
# Demo
If you want to run detection on your own images with a pre-trained model, download the pretrained model or train your own models at first, then add images to folder $ROOT/images, and then run
```
python demo.py --net vgg16 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir path/to/model/directoy
```
# Citation
```
@inproceedings{miccai,
  title={Positive-unlabeled Learning for Cell Detection in Histopathology Images with Incomplete Annotations},
  author={Zhao, Zipei and Pang, Fengqian and Liu, Zhiwen and Ye, Chuyang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={509--518},
  year={2021}
}
```
