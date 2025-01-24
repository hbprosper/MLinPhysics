#------------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# July 2021
#------------------------------------------------------------------------------
import os, sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#------------------------------------------------------------------------------
DELAY = 10 # seconds - interval between plot updates
LOG_SWITCH = 3
#------------------------------------------------------------------------------
# The loss file should be a simple text file with two columns of numbers:
#
#   train-losses,validation-losses
#     
def get_losses(loss_file):
    try:
        losses = pd.read_csv(loss_file).to_numpy()
        return losses[:, 0], losses[:, 1], losses[:, 2]
    except:
        return None

def get_timeleft(timeleft_file):
    try:
        return open(timeleft_file, 'r').read().strip()
    except:
        return None

class TimeLeft:
    '''
    Return the amount of time left.
    
    timeleft = TimeLeft(N)
    
    N: maximum loop count
    
      for i in timeleft:
          : :

    or
       timeleft(i, extra)
      
    '''
    def __init__(self, N):
        self.N = N        
        self.timenow = time.time
        self.start = self.timenow()
        self.str = ''
        
    def __del__(self):
        pass
    
    def __timestr(self, ii):
        # elapsed time since start
        elapsed = self.timenow() - self.start
        s = elapsed
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s = s - 60*m
        hh= h
        mm= m
        ss= s
        
        # time/loop
        count = ii+1
        t = elapsed / count
        f = 1/t
        
        # time left
        s = t * (self.N - count)
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s =  s - 60*m
        percent = 100 * count / self.N

        return "%10d|%6.2f%s|%2.2d:%2.2d:%2.2d/%2.2d:%2.2d:%2.2d|%6.1f it/s" % \
            (count, percent, '%', hh, mm, ss, h, m, s, f)
        
    def __iter__(self):
        
        for ii in range(self.N):
            
            if ii < self.N-1:
                print(f'\r{self.__timestr(ii):s}', end='')
            else: 
                print(f'\r{self.__timestr(ii):s}')
                
            yield ii
            
    def __call__(self, ii, extra='', colorize=False):
        
        if extra != '':
            if colorize:
               extra = "\x1b[1;34;48m|%s\x1b[0m" % extra
                
        self.a_str = f'{self.__timestr(ii):s}{extra:s}'
        
        if ii < self.N-1:
            print(f'\r{self.a_str}', end='')
        else:
            print(f'\r{self.a_str}')

    def __str__(self):
        return self.a_str

#--------------------------------------------------------------------
class Monitor:
    '''    
    monitor = Monitor()
        :   :
    monitor()
    '''
    def __init__(self, loss_file, timeleft_file):
        self.loss_file = loss_file
        self.timeleft_file = timeleft_file
        
        # set up an empty figure
        self.fig = plt.figure(figsize=(6, 4))
        self.fig.suptitle(loss_file)

        # add a subplot to it
        nrows, ncols, index = 1,1,1
        self.ax  = self.fig.add_subplot(nrows, ncols, index)
        
    def plot(self, frame):
        fig, ax = self.fig, self.ax
        
        ax.clear()
        ax.set_xlabel('iteration', fontsize=16)
        ax.set_ylabel('E[loss]', fontsize=16)
        ax.grid(True, which="both", linestyle='-')
        fig.tight_layout()
        
        data = get_losses(self.loss_file)
        
        if type(data) != type(None):
            
            iters, train_losses, valid_losses = data
            
            if len(train_losses) > 0:
                
                if train_losses[0]/train_losses[-1] > LOG_SWITCH:
                    ax.set_yscale('log')
        
                timeleft = get_timeleft(self.timeleft_file)
                if timeleft != None:
                    ax.set_title(timeleft, fontsize=9)
                else:
                    ax.set_title('iteration: %5d | %s' % (iters[-1], time.ctime()))
                    
                ax.plot(iters, train_losses, c='red',  label='training')
                ax.plot(iters, valid_losses, c='blue', label='validation')
                
                ax.legend()

    def __call__(self):        
        self.ani = FuncAnimation(fig=self.fig, 
                                 func=self.plot, 
                                 interval=1000*DELAY, # milliseconds
                                 repeat=False, 
                                 cache_frame_data=False)
        plt.show()

#--------------------------------------------------------------------
class LossWriter:
    '''
    Write training and validation losses to a csv file. The losses
    can be monitored while training by runnin the command

        python monitor_losses.py losses.csv&

    where losses.csv is the name of the loss file
    '''

    def __init__(self, model, niterations, 
                 lossfile, timeleftfile, paramsfile, 
                 step):

        # cache inputs
        self.model = model
        self.niterations = niterations
        self.lossfile = lossfile
        self.timeleftfile = timeleftfile
        self.paramsfile = paramsfile
        self.step = step
        
        # start saving model parameters after the following number of iterations.
        self.start_saving = niterations // 100
        self.min_avloss   = float('inf')  # initialize minimum average loss
    
        # initialize loss file
        # create loss file if it does not exist
        if not os.path.exists(lossfile):
            open(lossfile, 'w').write('iteration,t_loss,v_loss\n')  
    
        # get last iteration number from loss file
        df = pd.read_csv(lossfile)
        if len(df) < 1:
            self.itno = 0
        else:
            self.itno = df.iteration.iloc[-1] # get last iteration number

        self.timeleft = TimeLeft(niterations)
        
    def __call__(self, ii, t_loss, v_loss):

        # update loss file
        
        open(self.lossfile, 'a').write(f'{self.itno:12d},{t_loss:12.8},{v_loss:12.8}\n')

        # save model parameters to csv file
        
        if v_loss < self.min_avloss:
            self.min_avloss = v_loss
            if ii > self.start_saving:
                self.model.save(self.paramsfile)

        # update time left file
        
        line = f'|{self.itno:12d}|{t_loss:12.8f}|{v_loss:12.8f}|'
        self.timeleft(ii, line)
        open(self.timeleftfile, 'w').write(f'{str(self.timeleft):s}\n')

        # update iteration number
        
        self.itno += self.step