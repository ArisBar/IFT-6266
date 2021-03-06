# IFT-6266 Project

## Conditional Image Generation

### In a nutshell

**Goal**. The project is to generate the center region of images conditioned on the outside border and a caption. 
The task is applied on images from the mscoco dataset downsampled to 64 x 64 images, where the 32 x 32 center is masked out.

**Approach amd results**. I focused on conditioning on the outside border, leaving the inclusion of captions for future work. 
I approached the problem using deep convolutional autoencoder architectures akin to Ref [1], with L2 and adversarial losses [2]. 
I obtained my best results with a relatively simple fully convolutional architecture with skip connections, using batchnorm, the Adam optimizer, and a weighted combination of a L2  loss and a Wgan loss [3]. Images generated by my best model are shown below. 

#### Best Wgan: Training (left) and validation (right) images 
![Best_training](/images/Wgan_new_F5_train185.png)        !![Best_val](/images/Wgan_new_F5_val195.png)

Details of my experiments are given in the next sections.

**Implementation**.
I build my models by modifying the original DCGAN code [4] in core Theano. Note that the code uses the old cuda Theano backend.  


### Baseline: Autoencoder with L2 loss 
My baseline model for this project is an autoencoder (AE) with convolution and deconvolution layers and L2 loss reconstruction of the center.  The masked 64 x 64 image are  downsampled all the way down to 1 x 1;  and back to a 32 x 32 center for the image.  My 'deconv' layers here simply consists of upsampling followed a by standard convolution.  I've used leakyReLU activations for downsamling and ReLU ones for  upsampling and a L2 reconstruction loss for the centre image.   

I've experimented with variants of the following architecture:

| Layer | Input | Output | Kernel size |                 
| ------|-------|--------|-------------|
| conv1 | 3 x 64 x 64 | 64 x 32 x 32 | 3 x 3 |
| conv2 | 64 x 32 x 32 | 128 x 16 x 16 | 3 x 3 |
| conv3 |  128 x 16 x 16 | 256 x 8 x 8 | 3 x 3 |
| conv4 |  256 x 8 x 8 | 512 x 4 x 4 | 3 x 3 |
| conv4 | 512 x 4 x 4 | 512 x 1 x 1 | 4 x 4
| deconv1 | 512 x 1 x 1 | 256 x 4 x 4 | 3 x 3 |
| deconv2 |  256 x 4 x 4 | 128 x 8 x 8 | 3 x 3 |
| deconv3 | 128  x 8 x 8 | 64 x 4 x 4 | 3 x 3 |
| deconv4 | 128 x 16 x 16 | 3 x 32 x 32 | 3 x 3 |


Examples of images generated by this baseline are as follows: 

#### Autoencoder with L2 loss: Training (left) and validation (right) images                

![AE_training](/images/train195.png)   !![AE_validation](/images/val195.png) 

Although the performance of this baseline is not striking, we see that the model is already able to seome extent to extend edges, texture and colors from the border. 

I've experimented with variants including a channel-wise dense layer as in Ref [2] and a fully dense layer, without noticing significant improvements. Later, in the context of GAN, I would choose to lighten a bit this architecture by removing conv4 and deconv1, add skip connections to reinforce the context, reduce the number fo filters (max 256) and work with larger kernel size (5x5). I haven't tried to re-train my AE with such modifications but I believe it should lead to an improvement of the performance. 

### GAN architectures 

The various architectures I've tested evolved as follows:

- DCGAN: addition of a **discriminator** convolutional network with sigmoid output activation,  trained jointly with the generator to discriminate between fake and real images [2]; the generator is trained to fool the discrimonator. 
- Upgrade to Wgan loss, which arguably stabilizes training and addresses the mode collapse problem. The Wgan loss is obtained from standard GAN by removing the log of the adversarial losses, clipping the weights at each iteration throughout training, and using linear ouput activation for the discriminator (instead of a sigmoid). 


For the first version of these models, I've used as generator loss the sum of an adversarial loss and a L2 recontruction loss 
(my motivation was to guide the beginning of training to prevent mode collapse). As it turns out, I've observed that these two contributions were badly scaled, with an excessive  dominance of the L2 loss contribution. 
As a result the generated  images suspiciously looked like the ones I had obtained with my baseline autoencoder. 

Once the L2 term has been rescaled (renormalized by a factor alpha =e-5), I started observing typical GAN patters. 
However I've had a hard time  making my models converge. 
The following images were generated by one of my Wgan models, pretrained (for a few dozens epochs) with a L2 loss (before setting alpha = 0), where we clearly see typical mode collapse patterns: 

![NonConvWgan](/images/Wgan_L2_pretrain_val195.png)

As an attempt to overcome these difficulites I decided to simplify my models in several ways: 

- use less filters (max number of channels = 256)
- modify the generator to downsample only until 4 x 4  and up again

which slightly improved convergence and the quality of my images. However the key steps towards having a decent generative model have been the following: 

- add 'skip connections' between downsampling and upsampling layers in the generator (thanks are due to Alex Lamb and Sandeep Subramanian for suggesting the idea), which not only facilitates convergence but re-enforce the conditioning on the context. These are built by concatenating the corresponding layers along the channel axis. 
- I've increased the filter size to 5 x 5. That had a great impact on the image quality. 

I ended up with the following architecture for my Wgan:

#### Generator


| Layer | Input | Output | Kernel size |                 
| ------|-------|--------|-------------|
| conv1 | 3 x 64 x 64 | 32 x 32 x 32 | 5 x 5 |
| conv2 | 32 x 32 x 32 | 64 x 16 x 16 | 5 x 5 |
| conv3 |  64 x 16 x 16 | 128 x 8 x 8 | 5 x 5 |
| conv4 |  128 x 8 x 8 | 256 x 4 x 4 | 5 x 5 |
| deconv1| 256 x 4 x 4 | (128+128) x 8 x 8 | 5 x 5 |
| deconv2 | (128+128)  x 8 x 8 | (64+64) x 4 x 4 | 5 x 5 |
| deconv3 | (64+64) x 16 x 16 | 3 x 32 x 32 | 5 x 5 |

#### Discriminator 

| Layer | Input | Output | Kernel size |                 
| ------|-------|--------|-------------|
| conv1 | 3 x 64 x 64 | 32 x 32 x 32 | 5 x 5 |
| conv2 | 32 x 32 x 32 | 64 x 16 x 16 | 5 x 5 |
| conv3 |  64 x 16 x 16 | 128 x 8 x 8 | 5 x 5 |
| conv4 |  128 x 8 x 8 | 256 x 4 x 4 | 5 x 5 |
| pool + flatten | 256 x 4 x 4 | 256 ||
| output | 256 | 1 | | 


which generated the images shown in the Introduction. 


### Additional Experimentation 

I've continued experimenting with my model, by modifying it in two ways.

#### Adding noise 

The generator's input in traditional GAN are purely noisy samples. 
I was curious to see whether adding an additional noisy input in the middle of my generator 
(by downsampling the border down to 1 x 1, flatten, and concatenating to the resulting state a randomly generated noise vector before upsampling again) would improve the expressivity of the model.  Examples generated by the noisy model (taken here from the training images) are shown below. 

#### Wgan with additional noisy inputs: Training (left) and validation (right) 

![Wgan_noise](/images/Wgan_noise_train190.png)  !![Wgan_noise_val](/images/Wgan_noise_val195.png)

The advantage of this modification is not obvious.  In fact the validaiton images look slightly  worse -- more blurry and greyish.  

#### Reinforced context 

Finally, whereas it shoudl not be difficult to improve the image quality using a more approriate choice of hyperparameters, 
an important challenge is to insure continuity with the border to make the reconstruction more realistic.   I can think of two ways to (further) 'reinforce context'  and improve border continuity: 

- Re-inject the input (border) of the generator at each of the downsampling layers. One way to do this is to convolve the input image with 1 x 1 kernels  with the right number of filters and to concatenate the features maps. 
- Modify the generator to generate a thickening of the center image (say 36 x 36 instead of 32 x 32)  by a few pixels (trained with a L2 loss to match the corresponding region of the original image).

I've implemented the 2nd option, the results are as follow: 

#### Wgan with reinforced context: Training (left) and validation (right) images

![Wgan_context_train](/images/Wgan_context_train190.png)  !![Wgan_context_val](/images/Wgan_context_val195.png)



### Acknowledgments. 

Thanks are due to Stephanie Laroque, Sandeep Subramanian and Chiheb Trabelsi for discussions and help. 
An immense thank you also to Olexa Bilaniuk for his super effective technical support, especially debugging Theano. 


### References

[1] Context Encoders: Feature Learning by Inpainting.  https://arxiv.org/abs/1604.07379

[2] Generative adversarial Networks. https://arxiv.org/abs/1406.2661

[3] Wasserstein GAN.  https://arxiv.org/abs/1701.07875.

[4] Deep Convolutional Generative Adversarial Networks. Code available as  https://github.com/Newmu/dcgan_code



