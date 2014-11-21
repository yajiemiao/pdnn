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

class LearningRate(object):

    def __init__(self):
        '''constructor'''    
        
    def get_rate(self):
        pass

    def get_next_rate(self, current_error):
        pass

class LearningRateConstant(LearningRate):

    def __init__(self, learning_rate = 0.08, epoch_num = 20):

        self.learning_rate = learning_rate
        self.epoch = 1
        self.epoch_num = epoch_num
        self.rate = learning_rate

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):

        if ( self.epoch >=  self.epoch_num):
            self.rate = 0.0
        else:
            self.rate = self.learning_rate
        self.epoch += 1

        return self.rate

class LearningRateExpDecay(LearningRate):

    def __init__(self, start_rate = 0.08, scale_by = 0.5,
                 min_derror_decay_start = 0.05, min_derror_stop = 0.05, init_error = 100,
                 decay=False, min_epoch_decay_start=15, zero_rate = 0.0):

        self.start_rate = start_rate
        self.init_error = init_error
        
        self.rate = start_rate
        self.scale_by = scale_by
        self.min_derror_decay_start = min_derror_decay_start
        self.min_derror_stop = min_derror_stop
        self.lowest_error = init_error
        
        self.epoch = 1
        self.decay = decay
        self.zero_rate = zero_rate

        self.min_epoch_decay_start = min_epoch_decay_start


    def get_rate(self):
        return self.rate  
    
    def get_next_rate(self, current_error):
        diff_error = 0.0
        diff_error = self.lowest_error - current_error
            
        if (current_error < self.lowest_error):
            self.lowest_error = current_error
    
        if (self.decay):
            if (diff_error < self.min_derror_stop):
                self.rate = 0.0
            else:
                self.rate *= self.scale_by
        else:
            if ((diff_error < self.min_derror_decay_start) and (self.epoch > self.min_epoch_decay_start)):
                self.decay = True
                self.rate *= self.scale_by
            
        self.epoch += 1
        return self.rate


class LearningMinLrate(LearningRate):

    def __init__(self, start_rate = 0.08, scale_by = 0.5,
                 min_lrate_stop = 0.0002, init_error = 100,
                 decay=False, min_epoch_decay_start=15):

        self.start_rate = start_rate
        self.init_error = init_error

        self.rate = start_rate
        self.scale_by = scale_by
        self.max_epochs = max_epochs
        self.min_lrate_stop = min_lrate_stop
        self.lowest_error = init_error

        self.epoch = 1
        self.decay = decay
        self.min_epoch_decay_start = min_epoch_decay_start

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):
        diff_error = 0.0

        diff_error = self.lowest_error - current_error

        if (current_error < self.lowest_error):
            self.lowest_error = current_error

        if (self.decay):
            if (self.rate < self.min_lrate_stop):
                self.rate = 0.0
            else:
                self.rate *= self.scale_by
        else:
            if (self.epoch >= self.min_epoch_decay_start):
                self.decay = True
                self.rate *= self.scale_by

        self.epoch += 1
        return self.rate

# functions to save and resume the learning rate
# the following 4 fields are written into <lrate_file>, each field per line
# lrate.epoch: the current epoch
# lrate.rate: the current learning rate 
# lrate.lowest_error: the current lowest learning rate
# lrate.decay: whether decay has started
def save_lrate(lrate, lrate_file):
    file_open = open(lrate_file, 'w')  # always overwrite
    file_open.write(str(lrate.epoch) + '\n')
    file_open.write(str(lrate.rate) + '\n')
    file_open.write(str(lrate.lowest_error) + '\n')
    file_open.write(str(int(lrate.decay)) + '\n')
    file_open.close()

def resume_lrate(lrate, lrate_file):
    file_open = open(lrate_file, 'r')
    line = file_open.readline().replace('\n','')
    lrate.epoch = int(line)
    line = file_open.readline().replace('\n','')
    lrate.rate = float(line)
    line = file_open.readline().replace('\n','')
    lrate.lowest_error = float(line)
    line = file_open.readline().replace('\n','')
    lrate.decay = bool(int(line))
    file_open.close()



