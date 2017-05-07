# IFT-6266 Project

## Conditional Image Generation

### Introduction

**Goal**. The project is to generate the center region of images conditioned on the outside border and a caption. 
The task is applied on images from the mscoco dataset downsampled to 64 x 64 images, where the 32 x 32 center is masked out:

**My approach**. I focussed on conditioning on the outside border, leaving the inclusion of captions for future work. 
I approached the problem using deep convolutional autoencoder architectures with L2 and adversarial losses. 
I obtained my best results with a relatively simple fully convolutional architecture with skip connections, reLu and leaky reLu activations, and 5 x 5 kernel size, using a Wasserstein GAN (WGAN) loss. Exemples of reconstructed images from the mscoco  validation set are given below.



** Implementation** 
I build my models based on the original DCGAN code in core Theano. Note that the code uses the old cuda Theano backend.  

Details of my experiments are given in the next sections.

### Baseline: Autoencoder with L2 loss 



### Gan architectures 

### Reinforced context 

### Outlook

### Acknowledgments. 





