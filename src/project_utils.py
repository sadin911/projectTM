# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:50:23 2021

@author: chonlatid.d
"""

def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r',flush=True)