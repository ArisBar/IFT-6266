# %%
import sys
sys.path.append('..')

import pdb
import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
#debug = sys.gettrace() is not None
#if debug:
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

import theano.tensor as T
#from theano.sandbox.cuda.dnn import dnn_conv
#from theano.compile.debugmode import DebugMode


from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis, color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score

####################
#Load Data
###################
n_points = 100
#%%
from load import load_images
tr_set, val_set = load_images()


#%%
k = 1             # # of discrim updates for each gen update
l2 = 2.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngfc = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
nef = 64          # # of encod filters in first conv layer
ngf = 32          # # of decod filters in first conv layer
ndf = 32          # # of discrim filters in first conv layer
nbottleneck = 200
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
#beta = 1         # annealing of boundary L2 loss
ntrain, nval = len(tr_set), len(val_set),

def transform(X):
    return np.transpose(floatX(X)/255., (0,3,1,2))

def inverse_transform(X):
    X = np.transpose(X, (0,2,3,1))
    return X

def bound(X):
    return T.set_subtensor(X[:,:,16:48, 16:48], 0)

def stick(x, Y):
    return T.set_subtensor(Y[:,:,16:48, 16:48], x)



desc = 'contgan_code'
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

ew  = eifn((nef, nc, 3, 3), 'ew')
ew2 = eifn((nef*2, nef, 3, 3), 'ew2')
ew3 = eifn((nef*4, nef*2, 3, 3), 'ew3')
ew4 = eifn((nef*8, nef*4, 3, 3), 'ew4')
ewz = eifn((nbottleneck, nef*8, 4, 4), 'ewz')

gw  = gifn((ngf*8, nz, 3, 3), 'gw')
gw2 = gifn((ngf*4, ngf*8,  3, 3), 'gw2')
gw3 = gifn((ngf*2, ngf*4,   3, 3), 'gw3')
#gw4 = gifn((ngf*2, ngf,  3, 3), 'gw4')
gwx = gifn((nc,ngf*2, 3, 3), 'gwx')

dw  = difn((ndf, nc, 3, 3), 'dw')
dw2 = difn((ndf*2, ndf, 3, 3), 'dw2')
dw3 = difn((ndf*4, ndf*2, 3, 3), 'dw3')
dw4 = difn((ndf*8, ndf*4, 3, 3), 'dw4')
dw5 = difn((ndf*8*4*4, ndfc), 'dw5')
dwy = difn((ndfc, 1), 'dwy')

encod_params = [ew, ew2, ew3, ew4, ewz]
decod_params = [gw, gw2, gw3, gwx]
gen_params = encod_params + decod_params
discrim_params = [dw, dw2, dw3, dw4, dw5, dwy]


##
# Possible improvements
##
##################################
# Generator network
#################################
#%%


def gen(Y, ew, ew2, ew3, ew4, ewz, gw, gw2, gw3, gwx):

    # input is Y : (nc) x 64 x 64
    ##  Encoder: image --> noise
    h = lrelu(T.nnet.conv2d(Y, ew, subsample=(2, 2), border_mode=(1, 1)))
    # state size: (nef) x 32x 32
    h2 = lrelu(batchnorm(T.nnet.conv2d(h, ew2, subsample=(2, 2), border_mode=(1, 1))))
    # state size: (nef*2) x 16 x 16
    h3 = lrelu(batchnorm(T.nnet.conv2d(h2, ew3, subsample=(2, 2), border_mode=(1, 1))))
    # state size: (nef*4) x 8 x 8
    h4 = lrelu(batchnorm(T.nnet.conv2d(h3, ew4, subsample=(2, 2), border_mode=(1, 1))))
    # state size:  (nef*8) x 4 x4
    Z = lrelu(T.nnet.conv2d(h4, ewz))
    # final size: (nbottleneck) x 1 x 1

    #if z is not None:
    #    Z = add_noise(z, Z)

    ## Decoder: noise --> Image
    # input is Z: (nz) x 1 x 1
    hd = relu(batchnorm(deconv(Z, gw, ratio=4)))
    # state size: (ngf*8) x4 x4
    hd2 = relu(batchnorm(deconv(hd, gw2)))
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


def add_noise(Z, z):
    # add noise Z to downsampled image z
    z = T.flatten(z, 2)
    Z = T.concatenate([Z, z], axis=1)
    Z = Z.dimshuffle(0,1,'x','x')
    return Z


##################################
# Discriminator network
##################################

def discrim(X, w, w2, w3, w4, w5, wy):
    # input is X : (nc) x 64 x 64
    h = lrelu(T.nnet.conv2d(X, w, subsample=(2, 2), border_mode=(1, 1)))
    # state size: (ndf) x 32 x 32
    h2 = lrelu(batchnorm(T.nnet.conv2d(h, w2, subsample=(2, 2), border_mode=(1, 1))))
    # state size: (ndf*2) x 16 x 16
    h3 = lrelu(batchnorm(T.nnet.conv2d(h2, w3, subsample=(2, 2), border_mode=(1, 1))))
    # state size: (ndf*4) x 8 x 8
    h4 = lrelu(batchnorm(T.nnet.conv2d(h3, w4, subsample=(2, 2), border_mode=(1, 1))))
    # state size: (ndf*8) x 4 x 4
    h4 = T.flatten(h4, 2)
    # state size: ndf*8*4*4
    h5 = lrelu(batchnorm(T.dot(h4, w5)))
    # state size: ndfc
    y = sigmoid(T.dot(h5, wy))
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
gx = gen(bound(X), *gen_params)
gX = stick(gx, bound_X)

p_real = discrim(X, *discrim_params)
p_gen = discrim(gX, *discrim_params)

#print('gX', gX.test_value)

d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()
#g_cost_cond = ((bound_gX - Y)**2).sum()

#print('p_real', p_real.test_value, 'p_gen', p_gen.test_value)


#beta =sharedX(beta)
d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d
#g_cost = g_cost_d + beta*g_cost_cond

#print('d_cost', d_cost.test_value, 'g_cost', g_cost.test_value)

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

#print('cost', cost.test_value)

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates


## PROBLEM WITH COMPILATION, FIX THOSE FUCKING ADAM UPDATES!!!

# %%
print('COMPILING')
t = time()
_bound = theano.function([X], bound_X)
_train_g = theano.function([X], cost, updates=g_updates)
_train_d = theano.function([X], cost, updates=d_updates)
_gen = theano.function([X], gX)
print('%.2f seconds to compile theano functions'%(time()-t))


# %%
#################################
# Training
###############################

niter = 10
nbatch =10

print('TRAINING')
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(1, niter): #+niter_decay+1):
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
        print('g_cost=', g_cost , 'd_cost', d_cost)
    #    gX, gY = gen_samples(100000)
    #    gX = gX.reshape(len(gX), -1)
    #    va_nnc_acc_1k = nnc_score(gX[:1000], gY[:1000], vaX, vaY, metric='euclidean')
    #    va_nnc_acc_10k = nnc_score(gX[:10000], gY[:10000], vaX, vaY, metric='euclidean')
    #    va_nnc_acc_100k = nnc_score(gX[:100000], gY[:100000], vaX, vaY, metric='euclidean')
    #    va_nnd_1k = nnd_score(gX[:1000], vaX, metric='euclidean')
    #    va_nnd_10k = nnd_score(gX[:10000], vaX, metric='euclidean')
    #    va_nnd_100k = nnd_score(gX[:100000], vaX, metric='euclidean')
    #    log = [n_epochs, n_updates, n_examples, time()-t, va_nnc_acc_1k, va_nnc_acc_10k, va_nnc_acc_100k, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost]
    #    print('%.0f %.2f %.2f %.2f %.4f %.4f %.4f %.4f %.4f'%(epoch, va_nnc_acc_1k, va_nnc_acc_10k, va_nnc_acc_100k, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost))
    #    f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
    #    f_log.flush()

    if (epoch-1) % 1 == 0:
        sample_imb = transform(trX[0:25])
        sample_ymb = _bound(sample_imb)
        samples = np.asarray(_gen(sample_ymb))
        plt.imshow(color_grid_vis(inverse_transform(samples), 5, 5, 'samples/%s/%d.png'%(desc, n_epochs)))
        plt.show()
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
        joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, n_epochs))
