#!/usr/bin/env python
#------------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# July 2021
#------------------------------------------------------------------------------
import os, sys
import lossmonitor as lm
import time
#------------------------------------------------------------------------------
def main():
    # get name of loss file
    argv = sys.argv[1:]
    argc = len(argv)
    if argc < 1:
        sys.exit('''
        Usage:
           ./monitor_losses.py loss-file [timeleft-file]
    ''')
        
    loss_file = argv[0]
    print()
    print('loss file:     ', loss_file)
    
    if argc > 1:
        timeleft_file = argv[1]
        print('timeleft file: ', timeleft_file)
        print()
    else:
        timeleft_file = None

    monitor = lm.Monitor(loss_file, timeleft_file)
    
    monitor()

    print('\nbye from monitor_losses.py!\n')
#------------------------------------------------------------------------------
try:
    main()
except KeyboardInterrupt:
    print('\nciao!\n')

