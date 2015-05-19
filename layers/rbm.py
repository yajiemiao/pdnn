# Copyright 2013    Yajie Miao    Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import numpy, theano
import theano.tensor as T
import collections

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class RBM(object):
    """Bernoulli-bernoulli restricted Boltzmann machine (RBM)  """
    
    def __init__(self, input=None, n_visible=1024, n_hidden=1024,
                 W = None, hbias = None, vbias = None, numpy_rng = None,
                 theano_rng = None):

        self.type = 'fc'
               
        self.n_visible = n_visible
        self.n_hidden  = n_hidden

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None :
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if W is None :
            initial_W = numpy.asarray( numpy_rng.uniform(
                      low = -4*numpy.sqrt(6./(n_hidden+n_visible)),
                      high = 4*numpy.sqrt(6./(n_hidden+n_visible)),
                      size = (n_visible, n_hidden)),
                      dtype = theano.config.floatX)
            # shared variables for weights and biases
            W = theano.shared(value = initial_W, name = 'W')

        if hbias is None :
            # shared variable for hidden units bias
            hbias = theano.shared(value = numpy.zeros(n_hidden,
                                  dtype = theano.config.floatX), name='hbias')

        if vbias is None :
            # shared variable for visible units bias
            vbias = theano.shared(value = numpy.zeros(n_visible,
                                  dtype = theano.config.floatX), name='vbias')

        self.input = input
        if not input:
            self.input = T.matrix('input')
        
        self.delta_W = theano.shared(value=numpy.zeros_like(W.get_value(borrow=True), dtype=theano.config.floatX), name='delta_W')
        self.delta_hbias = theano.shared(value=numpy.zeros_like(hbias.get_value(borrow=True), dtype=theano.config.floatX), name='delta_hbias')
        self.delta_vbias = theano.shared(value=numpy.zeros_like(vbias.get_value(borrow=True), dtype=theano.config.floatX), name='delta_vbias')
        
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        
        # delta_parameters used with momentum
        self.delta_params = [self.delta_W, self.delta_hbias, self.delta_vbias]
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        ''' Compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1+T.exp(wx_b)),axis = 1)
        return -hidden_term - vbias_term 
    
    def propup(self, vis):
        ''' Propagate the visible activations up to the hidden units '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' Generates hidden unit outputs given visible inputs '''
        # the activation of the hidden units given visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        
        # assume that the hidden units will take sigmoid functions
        h1_sample = self.theano_rng.binomial(size = h1_mean.shape, n = 1, p = h1_mean,
                dtype = theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''Propagates the hidden activation downwards to the visible units'''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' Generates visible units given hidden units '''
        # the activation of the visible units given hiddens
        pre_sigmoid_v1, v1_mean  = self.propdown(h0_sample)
        
        # assume that the visible inputs are binary values
        v1_sample = self.theano_rng.binomial(size = v1_mean.shape,n = 1,p = v1_mean,
                dtype = theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' Gibbs sampling starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' Gibbs sampling starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]
        
    def get_cost_updates(self, batch_size = 128, lr = 0.0001, momentum=0.5, weight_cost=0.00001, persistent=None, k = 1):
        
        x, hp_data, h_data = self.sample_h_given_v(self.input)
        v_rec, v_rec_sigm, v_rec_sample = self.sample_v_given_h(h_data)
        a, hp_rec, b = self.sample_h_given_v(v_rec_sigm) 
                
        # gradient of parameters
        updates = collections.OrderedDict()
        updates[self.delta_W] = momentum * self.delta_W + lr * (1.0/batch_size) * (T.dot(self.input.T, hp_data) - T.dot(v_rec_sigm.T, hp_rec)) - lr * weight_cost * self.W
        updates[self.delta_hbias] = momentum * self.delta_hbias + lr * (1.0/batch_size) * (T.sum(h_data, axis=0) - T.sum(hp_rec, axis=0))
        updates[self.delta_vbias] = momentum * self.delta_vbias + lr * (1.0/batch_size) * (T.sum(self.input, axis=0) - T.sum(v_rec_sigm, axis=0))
        
        for param, dparam in zip(self.params, self.delta_params):
            updates[param] = param + updates[dparam]
        
        # approximation?? to free-energy cost 
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(v_rec_sample))
        # reconstruction cost
        monitoring_cost = T.mean(T.sqr(self.input-v_rec_sigm))
        
        return monitoring_cost, cost, updates

    def is_gbrbm(self):
        return False

class GBRBM(RBM):
    """Gaussian-bernoulli restricted Boltzmann machine"""
    
    def __init__(self, input=None, n_visible=351, n_hidden=1000,
                 W = None, hbias = None, vbias = None,
                 numpy_rng = None, theano_rng = None):
        
        super(GBRBM, self).__init__(input=input, n_visible=n_visible, n_hidden=n_hidden,
                    W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng, theano_rng=theano_rng)
    
    def free_energy(self, v_sample):
        ''' Compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5 * T.dot((v_sample - self.vbias), (v_sample - self.vbias).T)
        hidden_term = T.sum(T.log(1+T.exp(wx_b)), axis = 1)
        return -hidden_term - vbias_term
        
    def sample_v_given_h(self, h0_sample):
        ''' Generates visible units given hidden units '''
        # Compute the activation of the visible given the hiddens
        pre_sigmoid_v1, v1_mean  = self.propdown(h0_sample)
        v1_sample = self.theano_rng.normal(size = v1_mean.shape, avg=0.0, std=1.0, 
                         dtype = theano.config.floatX) + pre_sigmoid_v1

        return [pre_sigmoid_v1, v1_mean, v1_sample]
    
    def get_cost_updates(self, batch_size = 128, lr = 0.0001, momentum=0.5, weight_cost=0.00001, persistent=None, k = 1):
        
        x, hp_data, h_data = self.sample_h_given_v(self.input)
        v_rec, z, t = self.sample_v_given_h(h_data)
        a, hp_rec, b = self.sample_h_given_v(v_rec) #hid rec 
                
        updates = collections.OrderedDict()
        
        updates[self.delta_W] = momentum * self.delta_W + lr * (1.0/batch_size) * (T.dot(self.input.T, hp_data) - T.dot(v_rec.T, hp_rec)) - lr * weight_cost * self.W
        updates[self.delta_hbias] = momentum * self.delta_hbias + lr * (1.0/batch_size) * (T.sum(h_data, axis=0) - T.sum(hp_rec, axis=0))
        updates[self.delta_vbias] = momentum * self.delta_vbias + lr * (1.0/batch_size) * (T.sum(self.input, axis=0) - T.sum(v_rec, axis=0))
            
        updates[self.W] = self.W + updates[self.delta_W]
        updates[self.hbias] = self.hbias + updates[self.delta_hbias]
        updates[self.vbias] = self.vbias + updates[self.delta_vbias]
        
        # approximation?? to free-energy cost 
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(v_rec))
        # reconstruction cost
        monitoring_cost = T.mean(T.sqr(self.input - v_rec))
        
        return monitoring_cost, cost, updates
        

    def is_gbrbm(self):
        return True
