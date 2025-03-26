#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import matplotlib.pyplot as plt
from obspy.io.segy.segy import _read_segy, SEGYBinaryFileHeader
from obspy import read
import numpy as np
import pandas as pd
from collections import Counter
import random
import gc
gc.collect()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


# In[ ]:


filename = r''
parent_dir = os.path.dirname(os.path.abspath(filename))
grandparent_dir = os.path.dirname(parent_dir)


# # Read use _read_sgy()

# In[ ]:


# t0=time.time()
# segy = _read_segy(filename)
# print('--> data read in {:.1f} sec'.format(time.time()-t0))


# In[ ]:


# binary_file_header = segy.binary_file_header
# print("\nbinary_file_header:\n", binary_file_header)

# textual_file_header = segy.textual_file_header
# print("\ntextual_file_header:\n", textual_file_header)

# data_encoding=segy.data_encoding
# print("\ndata_encoding:\n",data_encoding)
# endian=segy.endian
# print("\nendian:\n", endian)
# file=segy.file
# print("\nfile:\n", file)
# classinfo = segy.__class__
# print("\nclassinfo:\n", classinfo)
# doc = segy.__doc__
# print("\ndoc:\n", doc)
# ntraces=len(segy.traces)
# print("\nntraces:\n", ntraces)
# size_M=segy.traces[0].data.nbytes/1024/1024.*ntraces
# print("\nsize:\n\t", size_M,"MB")
# print("\t", size_M/1024, "GB")


# # Read use read()

# In[ ]:


t0=time.time()
print('sgy use read:')
stream = read(filename, format='SEGY',unpack_trace_headers=True)
print('--> data read in {:.1f} min'.format((time.time()-t0)/60))


# In[ ]:


print(stream[0])


# In[ ]:


data = {}
k = 100
for i in range(20):
    trace_header = stream[k * 10 + i].stats.segy.trace_header
    for key, value in trace_header.items():
        if isinstance(value, (int, float)) and value != 0:
            if key not in data:
                data[key] = []
            data[key].append(value)
df = pd.DataFrame(data)
df = df.transpose()
df


# In[ ]:


il_name = 'for_3d_poststack_data_this_field_is_for_in_line_number'
xl_name = 'for_3d_poststack_data_this_field_is_for_cross_line_number'
il=[]
xl=[]
for i in range(len(stream)):
    trace_i_header = stream[i].stats.segy.trace_header
    il.append(trace_i_header[il_name])
    xl.append(trace_i_header[xl_name])


# In[ ]:


inlines = np.unique(il)
print(inlines)
print(len(inlines))


# In[ ]:


xlines = np.unique(xl)
print(xlines)
print(len(xlines))


# Check if the data is a cube shape

# In[ ]:


t0=time.time()
counter = Counter(il)
print('Count in {:.1f} sec'.format(time.time()-t0))
print (sorted(counter.items()))


# In[ ]:


t0=time.time()
counter = Counter(xl)
print('Count in {:.1f} sec'.format(time.time()-t0))
print (sorted(counter.items()))


# # this is a cube shape dataset.

# In[ ]:


del data,df,il,xl,counter
gc.collect()
seis_np = np.zeros((len(inlines),len(xlines),stream[0].stats.npts), dtype=np.float32)  # declare an empty array
print(seis_np.shape)


# In[ ]:


t0=time.time()
batch_size = 32
for i in range(0, len(stream), batch_size): # fill in the empty 3D array based on trace il number and crossline number.
    batch = stream[i:i + batch_size]
    for trace in batch:
        trace_il = trace.stats.segy.trace_header[il_name]
        trace_xl = trace.stats.segy.trace_header[xl_name]
        il_idx = int((trace_il - inlines[0]) / (inlines[1]-inlines[0]))
        xl_idx = int((trace_xl - xlines[0]) / (xlines[1]-xlines[0]))
        seis_np[il_idx][xl_idx] = trace.data # here 1001 is the initial inline number of the thebe dataset, will 851 is the initial crossline number
print('--> data read in {:.1f} min'.format((time.time()-t0)/60))


# In[ ]:


vmax= np.max(np.abs(seis_np)) / 4
plt.figure(figsize=(10,10))
plt.imshow(seis_np[10,:,:].transpose(), "seismic", vmax=vmax, vmin=-vmax) 


# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(seis_np[:,200,:].transpose(), "seismic",vmax=vmax, vmin=-vmax)


# In[ ]:


t0=time.time()
print('sgy save as npy:')
file_name = os.path.basename(filename)
new_file_name = os.path.splitext(file_name)[0] + '.npy'
np.save(new_file_name,seis_np)
print('--> data save in {:.1f} min'.format((time.time()-t0)/60))

