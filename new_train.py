#!/usr/bin/env python
# %%
import sys
sys.path.append('..')

import pdb
import os
import json
from time import time
import numpy as np
from tqdm import tqdm
#if 'DISPLAY' not in os.environ:
#    matplotlib.use('Pdf')
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import lasagne


import theano
#debug = sys.gettrace() is not None
#if debug:
#theano.config.optimizer='None'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

import theano.tensor as T
#from theano.sandbox.cuda.dnn import dnn_conv
#from theano.compile.debugmode import DebugMode


from lib import activations
from lib import updates
from lib.updates import clip_norms
from lib import inits
from lib.vis import grayscale_grid_vis, color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score

####################
#Load Data
####################
#%%
from load import load_images
tr_set, val_set = load_images()

# Select model
WGAN = False
WGAN_L2 = False
WGAN_L2boost = False
skip = True     # add skip connections
WGAN_new = True # smaller layers, no bottleneck
F5 = True       # augment filter size to 5
Noise = False
context = True

#%%
k = 1             # # of discrim updates for each gen update
if WGAN or WGAN_L2 or WGAN_new:      # in WGAN we train more the discriminator
    k=5
c = 0.01          # discriminatoor weight clipping factor in WGAN
l2 = 2.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
rho=0.5           # momnetum tern in RMSprop
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngfc = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
nef = 32
ngf = 32
if Noise:
    nef = 16
    ngf = 16
ndf = 32          # # of discrim filters in first conv layer
nbottleneck = 200
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100  # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
alpha = 0.00001   # weight of  L2 norm cost
#alpha = 1
#if WGAN_L2boost:
#    alpha = 0.001
#lr_rms = 0.00005  # initial learning rate for RMSprop
#beta = 1         # annealing of boundary L2 loss
ntrain, nval = len(tr_set), len(val_set),

def transform(X):
    return np.transpose(floatX(X)/255., (0,3,1,2))

def inverse_transform(X):
    X = np.transpose(X, (0,2,3,1))
    return X

def bound(X):
    return T.set_subtensor(X[:,:,16:48, 16:48], 0)

def center(X):
    return X[:,:,16:48, 16:48]

def center_thick(X):
    return X[:,:,2:34, 2:34]

def thick_bord(Y,k):
    return Y[:,:,(16-k):48+k, 16-k:48+k]

def stick(x, Y):
    return T.set_subtensor(Y[:,:,16:48, 16:48], x)




desc = 'gan'
if WGAN:
    desc= 'Wgan'
    if skip:
        desc ='Wgan_skip'
if WGAN_L2:
    desc = 'Wgan_L2'
    if skip:
        desc ='Wgan_L2_skip'
    if WGAN_L2boost:
        desc = 'WGAN_L2boost'

if WGAN_new:
    desc = 'WGAN_new'
    if skip:
        desc = 'Wgan_new_skip'
    if F5:  # set filter size to 5 x 5.
        desc = 'Wgan_new_F5'
        if Noise:
            desc= 'Wgan_new_F5_noise'
    if context:
        desc='Wgan_context'
#desc = 'Wgan_new_L2'



model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy

#####################################
# Initialize the parameters
#####################################
#%%
eifn = inits.Normal(scale=0.02)
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)

nF = 3
if F5:
    nF=5

ew  = eifn((nef, nc, nF, nF), 'ew')
ew2 = eifn((nef*2, nef, nF, nF), 'ew2')
ew3 = eifn((nef*4, nef*2, nF, nF), 'ew3')
ew4 = eifn((nef*8, nef*4, nF, nF), 'ew4')
#ewz = eifn((nbottleneck, nef*8, 4, 4), 'ewz')


#gw  = gifn((ngf*8, nz, 3, 3), 'gw')
gw2 = gifn((ngf*4, ngf*8,  nF, nF), 'gw2')
gw3 = gifn((ngf*2, ngf*4,   nF, nF), 'gw3')
#gw4 = gifn((ngf*2, ngf,  3, 3), 'gw4')
gwx = gifn((nc,ngf*2, nF, nF), 'gwx')

# weights for skip connections
#sgw  = gifn((ngf*8, nz, 3, 3), 'sgw')
sgw2 = gifn((ngf*4, ngf*8,  nF, nF), 'gw2')
sgw3 = gifn((ngf*2, (ngf+nef)*4,  nF, nF), 'sgw3')
#gw4 = gifn((ngf*2, ngf,  3, 3), 'gw4')
sgwx = gifn((nc,(ngf+nef)*2, nF, nF), 'sgwx')

if Noise:
    ew5 = eifn((nef*16, nef*8, nF, nF), 'ew5')
    ew6 = eifn(((nef)*16 *2 *2, nbottleneck), 'ew6')
    sgw1 = gifn((nbottleneck + nz, ngf*8*4*4), 'sgw1')



dw  = difn((ndf, nc, nF, nF), 'dw')
dw2 = difn((ndf*2, ndf, nF, nF), 'dw2')
dw3 = difn((ndf*4, ndf*2, nF, nF), 'dw3')
dw4 = difn((ndf*8, ndf*4, nF, nF), 'dw4')
#dw5 = difn((ndf*16, ndf*8, 3, 3), 'dw5')
#dw6 = difn((ndf*32, ndf*16, 2, 2), 'dw6')
#dw5 = difn((ndf*8*4*4, ndfc), 'dw5')
dwy = difn((ndf*8, 1), 'dwy')

#encod_params = [ew, ew2, ew3, ew4, ewz]
encod_params = [ew, ew2, ew3, ew4]
if skip:
    decod_params = [sgw2, sgw3, sgwx]
else:
    decod_params = [gw2, gw3, gwx]

gen_params = encod_params + decod_params
discrim_params = [dw, dw2, dw3, dw4, dwy]

if Noise:
    gen_params = [ew, ew2, ew3, ew4, ew5, ew6, sgw1, sgw2, sgw3, sgwx]




##
# Possible improvements
##
##################################
# Generator network
#################################
#%%

# Generator woth no skip connection
def gen_no_skip(Y, ew, ew2, ew3, ew4, gw2, gw3, gwx): #ewz, gw, gw2, gw3, gwx):
    # input is Y : (nc) x 64 x 64
    ##  Encoder: image --> noise
    h = lrelu(T.nnet.conv2d(Y, ew, subsample=(2, 2), border_mode='half'))
    # state size: (nef) x 32x 32
    h2 = lrelu(batchnorm(T.nnet.conv2d(h, ew2, subsample=(2, 2), border_mode='half')))
    # state size: (nef*2) x 16 x 16
    h3 = lrelu(batchnorm(T.nnet.conv2d(h2, ew3, subsample=(2, 2), border_mode='half')))
    # state size: (nef*4) x 8 x 8
    h4 = lrelu(batchnorm(T.nnet.conv2d(h3, ew4, subsample=(2, 2), border_mode='half')))
    # state size:  (nef*8) x 4 x4
    #if z is not None:
    #    Z = add_noise(z, Z)
    # state size: (ngf*8) x4 x4
    hd2 = relu(batchnorm(deconv(h4, gw2)))
    # state size : (ngf*4) x 8 x 8
    hd3 = relu(batchnorm(deconv(hd2, gw3)))
    # state size : (ngf*2)  x 16 x 16
    #####
    #hd4 = relu(batchnorm(deconv(hd3, gw4)))
    # state size: (ngf) x 16 x 16
    #####
    x = sigmoid(deconv(hd3, gwx))
    # output is x: (nc) x 32 x 32

    return x

# Generator with persistentcontext
def gen_skip(Y, ew, ew2, ew3, ew4, sgw2, sgw3, sgwx):
    # input is Y : (nc) x 64 x 64
    ##  Encoder: image --> noise

    h = lrelu(T.nnet.conv2d(Y, ew, subsample=(2, 2), border_mode='half'))
    # state size: (nef) x 32x 32
    #h_cont = reinforce_context(h, Y)
    h2 = lrelu(batchnorm(T.nnet.conv2d(h, ew2, subsample=(2, 2), border_mode='half')))
    # state size: (nef*2) x 16 x 16
    h3 = lrelu(batchnorm(T.nnet.conv2d(h2, ew3, subsample=(2, 2), border_mode='half')))
    # state size: (nef*4) x 8 x 8
    h4 = lrelu(batchnorm(T.nnet.conv2d(h3, ew4, subsample=(2, 2), border_mode='half')))
    # state size:  (nef*8) x 4 x4
    # state size : (ngf*8) x 4 x 4
    hd2 = relu(batchnorm(deconv(h4, sgw2)))
    # state size : (ngf*4) x 8 x 8
    hd2 = T.concatenate([hd2, h3], axis=1)
    # state size : (ngf+nef)*4 x 8 x 8
    hd3 = relu(batchnorm(deconv(hd2, sgw3)))
    # state size : (ngf)*2 x 16 x 16
    hd3 = T.concatenate([hd3, h2], axis=1)
    # state size : (ngf+nef)*2 x 16 x 16
    x = sigmoid(deconv(hd3, sgwx))
    # output is x: (nc) x 32 x 32
    return x

def gen_context(Y, ew, ew2, ew3, ew4, sgw2, sgw3, sgwx):
    # input is Y : (nc) x 64 x 64
    ##  Encoder: image --> noise

    h = lrelu(T.nnet.conv2d(Y, ew, subsample=(2, 2), border_mode='half'))
    # state size: (nef) x 32x 32
    #h_cont = reinforce_context(h, Y)
    h2 = lrelu(batchnorm(T.nnet.conv2d(h, ew2, subsample=(2, 2), border_mode='half')))
    # state size: (nef*2) x 16 x 16
    h3 = lrelu(batchnorm(T.nnet.conv2d(h2, ew3, subsample=(2, 2), border_mode='half')))
    # state size: (nef*4) x 8 x 8
    h4 = lrelu(batchnorm(T.nnet.conv2d(h3, ew4, subsample=(2, 2), border_mode='half')))
    # state size:  (nef*8) x 4 x4
    # state size : (ngf*8) x 4 x 4
    hd2 = relu(batchnorm(deconv(h4, sgw2)))
    # state size : (ngf*4) x 8 x 8
    hd2 = T.concatenate([hd2, h3], axis=1)
    # state size : (ngf+nef)*4 x 8 x 8
    hd3 = relu(batchnorm(deconv(hd2, sgw3)))
    # state size : (ngf)*2 x 16 x 16
    hd3 = T.concatenate([hd3, h2], axis=1)
    # state size : (ngf+nef)*2 x 16 x 16
    upsamp = T.nnet.abstract_conv.bilinear_upsampling(hd3, ratio=2, use_1D_kernel=False)
    x = sigmoid(T.nnet.conv2d(upsamp, sgwx, border_mode='full'))
    # output is x: (nc) x (32+nF-1) x (32+nF-1)
    return x

def gen(Y, ew, ew2, ew3, ew4, gw2, gw3, gwx):
    if skip and not context:
        print('pick the generator with skips')
        return gen_skip(Y, ew, ew2, ew3, ew4, gw2, gw3, gwx)
    if context:
        print('pick the generator with skip and context')
        return gen_context(Y, ew, ew2, ew3, ew4, gw2, gw3, gwx)
    else:
        return gen_no_skip(Y, ew, ew2, ew3, ew4, gw2, gw3, gwx)

def add_noise(Z, z):
    # add noise Z to downsampled image z
    z = T.flatten(z, 2)
    Z = T.concatenate([Z, z], axis=1)
    Z = Z.dimshuffle(0,1,'x','x')
    return Z

def gen_noise(Z, Y, ew, ew2, ew3, ew4, ew5, ew6, sgw1, sgw2, sgw3, sgwx):
    # input is Y : (nc) x 64 x 64
    ##  Encoder: image --> noise

    h = lrelu(T.nnet.conv2d(Y, ew, subsample=(2, 2), border_mode='half'))
    # state size: (nef) x 32x 32
    #h_cont = reinforce_context(h, Y)
    h2 = lrelu(batchnorm(T.nnet.conv2d(h, ew2, subsample=(2, 2), border_mode='half')))
    # state size: (nef*2) x 16 x 16
    h3 = lrelu(batchnorm(T.nnet.conv2d(h2, ew3, subsample=(2, 2), border_mode='half')))
    # state size: (nef*4) x 8 x 8
    h4 = lrelu(batchnorm(T.nnet.conv2d(h3, ew4, subsample=(2, 2), border_mode='half')))
    # state size:  (nef*8) x 4 x4
    h5 = lrelu(batchnorm(T.nnet.conv2d(h4, ew5, subsample=(2, 2), border_mode='half')))
    # state size: (nef)*16 x 2 x 2
    h5 = T.flatten(h5, 2)
    # state size: (nef)*16 *2 *2
    h6 = lrelu(batchnorm(T.dot(h5, ew6)))
    # state size: nbottleneck
    z = T.concatenate([h6, Z], axis=1)
    # staste size: nbottlneck+nz
    hd1 = relu(T.dot(z, sgw1))
    hd1 = hd1.reshape((hd1.shape[0], ngf*8, 4, 4))
    # state size : (ngf*8) x 4 x 4
    hd2 = relu(batchnorm(deconv(hd1, sgw2)))
    hd2 = T.concatenate([hd2, h3], axis=1)
    # state size : (ngf+nef)*4 x 8 x 8
    hd3 = relu(batchnorm(deconv(hd2, sgw3)))
    hd3 = T.concatenate([hd3, h2], axis=1)
    # state size : (ngf+nef)*2 x 16 x 16
    x = sigmoid(deconv(hd3, sgwx))
    return x









##################################
# Discriminator network
##################################

def discrim(X, w, w2, w3, w4, wy):
    # input is X : (nc) x 64 x 64
    h = lrelu(T.nnet.conv2d(X, w, subsample=(2, 2), border_mode=(2, 2)))
    # state size: (ndf) x 32 x 32
    h2 = lrelu(batchnorm(T.nnet.conv2d(h, w2, subsample=(2, 2), border_mode='half')))
    # state size: (ndf*2) x 16 x 16
    h3 = lrelu(batchnorm(T.nnet.conv2d(h2, w3, subsample=(2, 2), border_mode='half')))
    # state size: (ndf*4) x 8 x 8
    h4 = lrelu(batchnorm(T.nnet.conv2d(h3, w4, subsample=(2, 2), border_mode='half')))
    # state size: (ndf*8) x 4 x 4
    h5 = T.signal.pool.pool_2d(h4, ws=(4, 4), ignore_border=True)
    h6 = T.flatten(h5, 2)
    #state size: ndf*8
    if WGAN or WGAN_L2 or WGAN_new:
        y = T.dot(h6, wy)
    else:
        y = sigmoid(T.dot(h6, wy))
    # output: scalar
    return y


#################################
# Compilation
###############################
# %%



X = T.tensor4(dtype='float32')  # full image
#Y = T.tensor4(dtype='float32')  # holed image
#Z = T.matrix()  # noise
#X.tag.test_value = floatX(np.random.rand(2, 3, 64, 64))
#Y.tag.test_value = floatX(np.random.rand(2, 3, 64, 64))


bound_X = bound(X)
#center_X = center(X)
#thick_bord_X = thick_bord(X, nF-1)

if Noise:
    gx = gen_noise(Z, bound(X), *gen_params)
else:
    gx = gen(bound(X), *gen_params)

if context:
    center_gx = center_thick(gx)
    gX = stick(center_gx, bound_X)
else:
    gX = stick(gx, bound_X)

p_real = discrim(X, *discrim_params)
p_gen = discrim(gX, *discrim_params)

#print('gX', gX.test_value)

if WGAN or WGAN_L2 or WGAN_new:
    d_cost_real = - p_real.mean()
    d_cost_gen  =   p_gen.mean()
    g_cost_d    = - d_cost_gen
else:
    d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen  = bce(p_gen,  T.zeros(p_gen.shape)).mean()
    g_cost_d    = bce(p_gen,  T.ones(p_gen.shape)).mean()

if context:
    g_cost_L2 = ((gx - thick_bord(X, 2))**2).sum()
else:
    g_cost_L2 = ((gx - center(X))**2).sum()
#g_cost_cond = ((bound_gX - Y)**2).sum()

#print('p_real', p_real.test_value, 'p_gen', p_gen.test_value)

if WGAN:
    alpha=0
#beta =sharedX(beta)
d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d + alpha*g_cost_L2


#g_cost = g_cost_d + beta*g_cost_cond





#print('d_cost', d_cost.test_value, 'g_cost', g_cost.test_value)

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

#print('cost', cost.test_value)

lrt = sharedX(lr)
alph = sharedX(alpha)
#lrt_rms = sharedX(lr_rms)


if WGAN or WGAN_L2 or WGAN_new:
    d_updater = updates.Adam_clip(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    g_updater = updates.Adam_clip(lr=lrt ,b1=b1,  regularizer=updates.Regularizer(l2=l2))
else:
    d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))


d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates


## PROBLEM WITH COMPILATION, FIX THOSE FUCKING ADAM UPDATES!!!

# %%
print('COMPILING')
t = time()
#_bound = theano.function([X], bound_X)
_train_g = theano.function([X], cost, updates=g_updates)
_train_d = theano.function([X], cost, updates=d_updates)
_gen = theano.function([X], gX)
print('%.2f seconds to compile theano functions'%(time()-t))

#sample_zmb = floatX(np_rng.uniform(-1., 1., size=(25, nz)))

f_log=open('logs/%s.ndjson'%desc,'wb')
log_fields = [
    'n_epochs',
    'n_updates',
    'n_examples',
    'time',
    'g_cost',
    'd_cost',
    'g_cost_d'
]

# %%
#################################
# Training
###############################

#niter = 20
#niter_decay = 0
#nbatch = 50


print('TRAINING',  desc.upper())
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(1, niter+niter_decay+1):
    print('epoch', epoch)
    trX = shuffle(tr_set)
    for imb in tqdm(iter_data(trX, size=nbatch), total=ntrain/nbatch):
        imb = transform(imb)
        #ymb = _bound(imb)
        #zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        if n_updates % (k+1) == 0:
            cost = _train_g(imb)
        else:
            cost = _train_d(imb)

        n_updates += 1
        n_examples += len(imb)
    if (epoch-1) % 5 == 0:
        g_cost = float(cost[0])
        d_cost = float(cost[1])
        g_cost_d = float(cost[2])
        #print('gen cost=', g_cost, "L2_loss cost=", g_cost-g_cost_d,  'discrimin cost=', d_cost)
    #    gX, gY = gen_samples(100000)
    #    gX = gX.reshape(len(gX), -1)
    #    va_nnc_acc_1k = nnc_score(gX[:1000], gY[:1000], vaX, vaY, metric='euclidean')
    #    va_nnc_acc_10k = nnc_score(gX[:10000], gY[:10000], vaX, vaY, metric='euclidean')
    #    va_nnc_acc_100k = nnc_score(gX[:100000], gY[:100000], vaX, vaY, metric='euclidean')
    #    va_nnd_1k = nnd_score(gX[:1000], vaX, metric='euclidean')
    #    va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost))
        log=[n_epochs, n_updates, n_examples, time()-t, g_cost, d_cost, g_cost_d]
        f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
        f_log.flush()

        sample_imb = transform(trX[0:25])
        sample_imb_val = transform(val_set[0:25])
        #sample_ymb = _bound(sample_imb)
        #sample_ymb_val = _bound(sample_imb_val)
        samples_train = np.asarray(_gen(sample_imb))
        samples_val = np.asarray(_gen(sample_imb_val))
        color_grid_vis(inverse_transform(samples_train), 5, 5, 'samples/%s/train%d.png'%(desc, n_epochs))
        color_grid_vis(inverse_transform(samples_val), 5, 5, 'samples/%s/val%d.png'%(desc, n_epochs))
        #plt.show()
    n_epochs += 1


    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
        joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, n_epochs))
