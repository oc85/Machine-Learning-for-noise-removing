# Machine-Learning-for-noise-removing



Pix2pix machine learning algorythm was used to perform noise removal and enhancement from confocal images having low resolution. The confocal images with high resolution were used as reference.
In order to speed up training, patchification was applied to the images with size 1200 x 1200 pixels.

In order to improve the effectivenes of the pix2pix the self-attentinal mechanism was added.

Training was performed on 256 images (without augmentation) during 5000 iterations on GPU:

![image](https://github.com/user-attachments/assets/a158a0a6-5a19-4d44-be50-b70e4a55a808)

