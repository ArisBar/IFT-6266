# IFT-6266 Project

## Conditional Image Generation

### In a nutshell

**Goal**. The project is to generate the center region of images conditioned on the outside border and a caption. 
The task is applied on images from the mscoco dataset downsampled to 64 x 64 images, where the 32 x 32 center is masked out:

**My approach**. I focussed on conditioning on the outside border, leaving the inclusion of captions for future work. 
I approached the problem using deep convolutional [1] autoencoder architectures akin to Ref [2], with L2 and adversarial losses. 
I obtained my best results with a relatively simple fully convolutional architecture with skip connections and a weighted combination of a L2  loss and a Wasserstein GAN (WGAN) loss [3]. Exemples of reconstructed images from the mscoco  validation set are given below.


Details of my experiments are given in the next sections.

**Implementation** 
I build my models based on the original DCGAN code in core Theano. Note that the code uses the old cuda Theano backend.  

**References**
[1] Context Encoders: Feature Learning by Inpainting. D. Pathak, P. Krähenbühl, J. Donahue

T. Darrell, A. A. Efros. Computer Vision and Pattern Recognition (CVPR), 2016.

[2] Deep Convolutional Generative Adversarial Networks, A.Radford, L. Metz, S.Chintala, code available as  https://github.com/Newmu/dcgan_code

[3] Wasserstein GAN, M. Arjovsky, S. Chintala, L. Bottou https://arxiv.org/abs/1701.07875.


### Baseline: Autoencoder with L2 loss 



### GAN architectures 

### Reinforced context 

### Outlook

### Acknowledgments. 

### References







