#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx 
import os, glob, time
import pandas as pd
import numpy as np
import pickle, random
import seaborn as sns

from matplotlib import pyplot as plt

# RDkit
#from rdkit import Chem
#from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# Pytorch and Pytorch Geometric
#import torch
#from torch_geometric.data import Data
#from torch.utils.data import DataLoader

#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem import Draw
#IPythonConsole.ipython_useSVG=True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import tensorflow as tf


# In[3]:


#!pip install tensorflow


# In[4]:


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE


# In[5]:


#!pip install tensorboard


# In[6]:


def multi_density(x, mean_1=-2, mean_2=2, sd_1=0.3, sd_2=0.6):
  from scipy.stats import norm
  return 0.5 * norm.pdf(x, loc=mean_1, scale=sd_1) + 0.5 * norm.pdf(x, loc=mean_2, scale=sd_2) 


# In[7]:



x_vec = np.linspace(-5, 5, 100)
y_vec = np.array([multi_density(x) for x in x_vec])
y_vec_Z = np.sum(y_vec)
y_vec = y_vec/y_vec_Z


# In[8]:


plt.plot(x_vec, y_vec)


# In[9]:


prob_matrix = np.zeros([100,100]) 
stay = 0.0
mean_direction = 1/2.0
for i in range(100):
  if i == 0:
    prob_matrix[i,i] = stay * 2
    prob_matrix[i,i+1] = 1.0-(stay*2)
  elif i == 99:
    prob_matrix[i,i] = stay*2
    prob_matrix[i,i-1] = 1.0-(stay*2)
  elif i == 50:
    prob_matrix[i,i] = stay
    prob_matrix[i,i-1] = (1.0-stay)*0.5
    prob_matrix[i,i+1] = (1.0-stay)*0.5
  elif i < 50:
    prob_matrix[i,i] = stay
    prob_matrix[i,i-1] = (1.0-stay)*(1.0-mean_direction)
    prob_matrix[i,i+1] = (1.0-stay)*(mean_direction)
  elif i > 50: 
    prob_matrix[i,i] = stay
    prob_matrix[i,i-1] = (1.0-stay)*(mean_direction)
    prob_matrix[i,i+1] = (1.0-stay)*(1.0-mean_direction)
  else:
    assert(False)
#prob_matrix = prob_matrix.transpose()
prob_matrix

np.sum(prob_matrix, axis=1)


# In[10]:


#np.sum(prob_matrix, axis=1)


# In[11]:


prob_matrix.shape


# In[12]:


#y_vec.shape
pi = y_vec.reshape(1,100) # 1,100


# In[13]:


TIME_SCALE = 1000
sol = np.zeros([TIME_SCALE,100])
for i in range(TIME_SCALE):
  # (n,k),(k,m)->(n,m)     n=100, k=100, m=1
  sol[i,:] = pi.flatten()
  pi = np.matmul(pi, prob_matrix)  # pi prob matrix


# In[14]:


plt.plot(x_vec, sol[0,:])
plt.plot(x_vec, sol[10,:])
plt.plot(x_vec, sol[300,:])
plt.plot(x_vec, sol[500-50,:])
plt.plot(x_vec, sol[TIME_SCALE-1,:])
#plt.xlim((0,250))
plt.ylim((0.0,np.max(sol)*1.1))


# In[15]:


pi_back_init = sol[-1,:].flatten()


# In[16]:


plt.plot(x_vec, pi_back_init)
plt.ylim((0.0,np.max(pi_back_init)*1.1))


# In[17]:


#!rm -rf output_kernel1


# In[18]:


sol.shape


# ### Gif

# In[19]:


os.system('mkdir output_kernel1')
for i in range(sol.shape[0]):
  if i == 0 or i == range(sol.shape[0])[-1] or i%10 == 0:
    plt.clf()
    plt.plot(x_vec, sol[i,:].flatten())
    plt.ylim((0.0,np.max(sol)*1.1)) 
    plt.title((str(i).zfill(5)))
    plt.savefig('output_kernel1/step_{}.png'.format(str(i).zfill(5)))
    #if i > 500:
    #  break


# In[20]:


import imageio
with imageio.get_writer('kernel1.gif', mode='I', fps=5.0) as writer:
    for filename in sorted(glob.glob('output_kernel1/step*.png')):
        image = imageio.imread(filename)
        writer.append_data(image)


# In[21]:


get_ipython().system('pip install pygifsicle')


# In[22]:


#!sudo apt-get install gifsicle && pip install pygifsicle


# In[23]:


time.sleep(1)
from pygifsicle import optimize
optimize("kernel1.gif")


# In[24]:


#def gen_for_kernel(name, kernel, size=100):
#  prob_matrix = np.zeros([size,size]) 
#  epsilon = 0.01
#  for i in range(size):
#    for j in range(size):
#      prob_matrix[i,j] = kernel(i,j) 



# # Sampling

# In[25]:


plt.plot(x_vec, sol[0,:])
plt.plot(x_vec, sol[-1,:])


# In[26]:


prior_vec = sol[-1,:].flatten()
prior_vec


# ### Sample Forward

# In[27]:


def gen_forward_trajectory():
  init_point = np.random.choice(list(range(100)), size=1, p=y_vec/np.sum(y_vec))
  init_v = int(init_point[0])
  trajetory = np.zeros(TIME_SCALE, dtype=int)
  trajetory[0] = init_v
  for i in range(TIME_SCALE-1):
    current_pos = trajetory[i]
    prob_go_to_left = 0.0
    try:
      prob_go_to_left = prob_matrix[current_pos, current_pos-1]
    except:
      pass
    prob_go_to_right = 0.0
    try:
      prob_go_to_right = prob_matrix[current_pos, current_pos+1]
    except:
      pass
    prob_stay = prob_matrix[current_pos, current_pos]
    next_state_prob = np.array([prob_go_to_left, prob_stay, prob_go_to_right])
    next_state_prob = next_state_prob / np.sum(next_state_prob)
    next_state = np.random.choice([-1,0,1], size=1, p=next_state_prob) +  trajetory[i]
    trajetory[i+1] = next_state
  return trajetory


# In[28]:


trajetories_forward = [gen_forward_trajectory() for _ in range(1000)]


# In[29]:


for trajetory in trajetories_forward:
  plt.plot(trajetory, alpha=0.02)
z = plt.ylim((0, 100))


# In[30]:


sample_data = list()
for traj in trajetories_forward:
  for i in range(len(traj)-1):
    input = torch.tensor([traj[i+1],i+1], dtype=torch.float, device=DEVICE)
    prediction = torch.zeros(3, dtype=torch.float, device=DEVICE)
    if traj[i+1] ==  traj[i]:
      prediction[1] = 1.0
    elif traj[i+1] ==  traj[i]+1:
      prediction[0] = 1.0
    elif traj[i+1] ==  traj[i]-1:
      prediction[2] = 1.0
    sample = (input, prediction)
    sample_data.append(sample) 

random.shuffle(sample_data)
sample_data[:10]


# In[31]:


def gen_trainloader(traj_num = 1000, mini_batch_size = 100):
    trajetories_forward = [gen_forward_trajectory() for _ in range(traj_num)]
    sample_data = list()
    for traj in trajetories_forward:
        for i in range(len(traj)-1):
            input = torch.tensor([traj[i+1],i+1], dtype=torch.float, device=DEVICE)
            prediction = torch.zeros(3, dtype=torch.float, device=DEVICE)
            if traj[i+1] ==  traj[i]:
            prediction[1] = 1.0
            elif traj[i+1] ==  traj[i]+1:
            prediction[0] = 1.0
            elif traj[i+1] ==  traj[i]-1:
            prediction[2] = 1.0
            sample = (input, prediction)
            sample_data.append(sample) 
    train_data = DataLoader(sample_data, batch_size=mini_batch_size, shuffle=True) 
    return train_data


# ### Sample Backward

# In[32]:


def gen_trajectory():
  init_point = np.random.choice(list(range(100)), size=1, p=prior_vec/np.sum(prior_vec))
  init_v = int(init_point[0])
  
  trajetory = np.zeros(TIME_SCALE, dtype=int)
  trajetory[0] = init_v
  for i in range(TIME_SCALE-1):
    current_pos = trajetory[i]

    prob_go_to_left = 0.0
    if current_pos != 0:
      prob_go_to_left = sol[TIME_SCALE-i-2, current_pos-1] * prob_matrix[current_pos-1, current_pos] # prior to be in left state at t-1 * prob to go from left state to current state

    prob_go_to_right = 0.0
    if current_pos != 99:
      prob_go_to_right = sol[TIME_SCALE-i-2, current_pos+1] * prob_matrix[current_pos+1, current_pos] # prior to be in left state at t-1 * prob to go from left state to current state

    prob_form_same =  sol[TIME_SCALE-i-2, current_pos] * prob_matrix[current_pos, current_pos]

    next_state_prob = np.array([prob_go_to_left, prob_form_same, prob_go_to_right])
    next_state_prob = next_state_prob / np.sum(next_state_prob)
    next_state = np.random.choice([-1,0,1], size=1, p=next_state_prob) +  trajetory[i]
    trajetory[i+1] = next_state
  return trajetory


# In[33]:


trajetories = [gen_trajectory() for _ in range(1000)]


# In[34]:


for trajetory in trajetories:
  plt.plot(trajetory, alpha=0.02)
z = plt.ylim((0, 100))


# #### plot sample map

# In[35]:


sample_map = np.zeros([TIME_SCALE,100])

for i in range(len(sample_data)):
  input, prediction = sample_data[i]
  state, t = input
  #print(state.detach().cpu().numpy(), t, prediction)
  if prediction[0].detach().cpu().numpy() == 1:
    sample_map[int(t.detach().cpu().numpy()), int(state.detach().cpu().numpy())] += -1
  elif prediction[2].detach().cpu().numpy() == 1:
    sample_map[int(t.detach().cpu().numpy()), int(state.detach().cpu().numpy())] += 1

#sample_map[0,0] = 300
#sample_map[0,1] = -300


# In[36]:


plt.imshow(sample_map, cmap='seismic', interpolation="nearest")
#plt.colorbar()
plt.savefig('map_sample.jpg', dpi=300)
plt.show()


# In[37]:


sns.displot([t[-1] for t in trajetories], kde=True)
z = plt.xlim((0, 100))


# In[38]:


os.system('mkdir output_back1')
for i in range(TIME_SCALE):
  if i == 0 or i == range(TIME_SCALE)[-1] or i % 10 == 0:
    plt.clf()
    sns.histplot([t[i] for t in trajetories], bins=20)
    #plt.plot(x_vec, sol[i,:].flatten())
    #plt.ylim((0.0,np.max(sol)*1.1)) 
    plt.xlim((0, 100))
    #plt.ylim((0, 0.12))
    plt.title((str(i).zfill(5)))
    plt.savefig('output_back1/step_{}.png'.format(str(i).zfill(5)))
    plt.close()


# In[39]:


import imageio
with imageio.get_writer('back1.gif', mode='I', fps=5.0) as writer:
    for filename in sorted(glob.glob('output_back1/step*.png')):
        image = imageio.imread(filename)
        writer.append_data(image)


# In[40]:


time.sleep(1)
from pygifsicle import optimize
optimize("back1.gif")


# In[ ]:





# ## Training

# In[86]:


class GuidanceNet(torch.nn.Module):
  def __init__(self, input_size=2, hidden_size=100, layer_number=15, out_size=3, softmax_out = True):
    super(GuidanceNet, self).__init__()
    self.input_size = input_size
    self.hidden_size  = hidden_size
    self.softmax = nn.Softmax(dim=1)
    self.softmax_out = softmax_out

    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU()) # or make leaky
    for i in range(layer_number):
      layers.append(nn.Linear(hidden_size, hidden_size))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(p=0.05))
    layers.append(nn.Linear(hidden_size, out_size))
    self.mlp = nn.Sequential(*layers)

  def forward(self, x):
    x = self.mlp(x)
    #x = x**2
    if not self.softmax_out:
      return x
    x = self.softmax(x)
    return x

#net = GuidanceNet()


# In[87]:


#input, target = sample_data[0]
#input = input.reshape([1,-1])
#target = target.reshape([1,-1])
#prediction = net(input)
#net(input), input, target


# In[88]:


#loss = nn.CrossEntropyLoss()
#loss(target, prediction)


# ### Evaluate Model

# In[89]:


def gen_trajectory_learned():
    init_point = np.random.choice(list(range(100)), size=1, p=[1/100.0 for _ in range(100)])
    init_v = int(init_point[0])

    trajetory = np.zeros(TIME_SCALE, dtype=int)
    trajetory[0] = init_v
    for i in range(TIME_SCALE-1):
        current_pos = trajetory[i]
        input = torch.tensor([current_pos, TIME_SCALE-i-1], dtype=torch.float, device=DEVICE)
        input = input.reshape(1, -1)
        prediction = model(input).flatten().detach().cpu().numpy()
        print(np.sum(prediction), prediction)
        assert(np.sum(prediction) > 0.9999 and np.sum(prediction) < 1.0001)

        next_state = np.random.choice([-1,0,1], size=1, p=prediction) +  trajetory[i]
        trajetory[i+1] = next_state
        trajetory[i+1] = max(0,trajetory[i+1])
        trajetory[i+1] = min(99,trajetory[i+1])
    return trajetory

def eval_model(model, epoch):
    model.eval()
    
    # map
    pred_map = np.zeros([TIME_SCALE,100])
    for t in range(TIME_SCALE):
        for pos in range(100):
            input = torch.tensor([pos, t], dtype=torch.float, device=DEVICE)
            input = input.reshape(1, -1) # batch format
            net_output = model(input)
            x = float(net_output[0][0] - net_output[0][2])
            pred_map[t, pos] = 1/(1+np.exp(-x))  #cut off at 0
    pred_map[0][0] = 1.0
    pred_map[0][1] = 0.0
    plt.clf()
    plt.imshow(pred_map, cmap='seismic', interpolation="nearest")
    plt.colorbar()
    plt.savefig('map_{}.png'.format(100000+epoch), dpi=300)
    
    
    #trajs 
    plt.clf()
    traj_learned = [gen_trajectory_learned() for _ in range(30)]
    for trajetory in traj_learned:
        plt.plot(trajetory, alpha=0.1)
    z = plt.ylim((0, 100))
    plt.savefig('trajs_learned_{}.png'.format(100000+epoch), dpi=300)
    plt.clf()
    
    #final
    traj_learned = [gen_trajectory_learned() for _ in range(500)]
    for pos in [0,-1,int(TIME_SCALE/2)]:
        final_state = [traj[pos] for traj in traj_learned]
        sns.displot(final_state, kde=True)
        z = plt.xlim((0, 100))
        plt.savefig('trajs_learned_final_{}_{}.png'.format(100000+epoch, pos), dpi=300)
        plt.clf()
    


# ## Training Procedure

# In[90]:


def train(train_loader, criterion, optimizer, model):
  model.train()
  loss_list = list()
  pbar = tf.keras.utils.Progbar(target=len(train_loader))
  for idx, (input, target) in enumerate(train_loader):
    optimizer.zero_grad() 
    input_noise = input #+ torch.randn(input.shape, device=DEVICE) * 0.0
    prediction = model(input_noise)
    loss = criterion(prediction, target)
    loss.backward()  
    optimizer.step()   
    loss = float(loss)
    loss_list.append(loss)
    pbar.update(idx, values=[("loss",loss)])
  return np.mean(loss_list)


# In[91]:


#for img in glob.glob('*.png'):
#    os.system('rm '+img)


# In[92]:


def loss_fn(prediction, target):
  loss_value = prediction*target
  loss_value = torch.sum(loss_value, dim=1)
  loss_value = -torch.log(loss_value+0.000000001) 
  loss_value = torch.mean(loss_value) # or sum for actual negative LL
  return loss_value


# In[93]:


prediction_1 = torch.tensor([[0.6, 0.4]], dtype=torch.float, device=DEVICE)
prediction_2 = torch.tensor([[0.001, 0.999]], dtype=torch.float, device=DEVICE)
target_1 = torch.tensor([[1.0, 0.0]], dtype=torch.float, device=DEVICE)
target_2 = torch.tensor([[0.0, 1.0]], dtype=torch.float, device=DEVICE)
print(loss_fn(prediction_1, target_1) + loss_fn(prediction_1, target_2) )
print(loss_fn(prediction_2, target_1) + loss_fn(prediction_2, target_2) )


# In[ ]:


# setup
criterion = loss_fn# nn.CrossEntropyLoss()
#criterion =  nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
model = GuidanceNet()
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
epoch_num = 10000
loss_list = list()
BATCH_SIZE = 5000


# data
cut = int(len(sample_data)*0.8)
#data_train = sample_data[:cut]
#data_test = sample_data[cut:]
#train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True) 
#test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False) 

train_loader = None

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/molecules')

# training loop
for epoch_i in range(epoch_num):
  if epoch_i % 50 == 0: # generate new samples after 10 epochs
    train_loader = gen_trainloader(traj_num = 5000, mini_batch_size = BATCH_SIZE)
  loss = train(train_loader, criterion, optimizer, model)
  loss_list.append(loss)
  print("   Epoch {}/{} finished.".format(epoch_i+1, epoch_num))
  writer.add_scalar("Loss/train", loss, epoch_i)
  writer.close()
  if epoch_i % 20 == 0:
    eval_model(model, epoch_i)

print('done')


# In[ ]:


pred_map = np.zeros([TIME_SCALE,100])
model.eval()

for t in range(TIME_SCALE):
  for pos in range(100):
    input = torch.tensor([pos, t], dtype=torch.float, device=DEVICE)
    input = input.reshape(1, -1) # batch format
    net_output = model(input)
    #print(net_output.shape)
    x = float(net_output[0][0] - net_output[0][2])
    pred_map[t, pos] = 1/(1+np.exp(-x))  #cut off at 0


# In[ ]:


pred_map[0][0] = 1.0
pred_map[0][1] = 0.0


# In[ ]:


plt.imshow(pred_map, cmap='seismic', interpolation="nearest")
#plt.set_clim(0.0,1.0)
plt.colorbar()
plt.savefig('map.png', dpi=300)
plt.show()


# In[ ]:


input = torch.tensor([50, 3], dtype=torch.float, device=DEVICE)
input = input.reshape(1, -1) # batch format
net_output = model(input)
net_output


# ### Sample from Map

# In[ ]:





# In[ ]:


traj_learned = [gen_trajectory_learned() for _ in range(100)]


# In[ ]:


for trajetory in traj_learned:
  plt.plot(trajetory, alpha=0.1)
z = plt.ylim((0, 100))


# In[ ]:


final_state = [traj[-1] for traj in traj_learned]
sns.displot(final_state, kde=True)
z = plt.xlim((0, 100))


# # Tensorboard

# In[ ]:


#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/molecules')


# In[ ]:


#writer.add_scalar("Loss/train", 3.0, 1)
#writer.add_scalar("Loss/train", 2.0, 2)
#writer.add_scalar("Loss/train", 1.5, 3)
#writer.flush()
#writer.close()


# In[ ]:


#%load_ext tensorboard


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir runs')


# In[ ]:


# from https://apoorvnandan.github.io/2020/10/27/keras-progress-bar-pytorch/


dataloader = [1,2,3]
def train_step(x):
  return random.random()

def validate():
  return random.random()

import tensorflow as tf

n_epochs = 3
for epoch in range(n_epochs):
    n_batches = len(dataloader)
    print(f'Epoch {epoch+1}/{n_epochs}')
    pbar = tf.keras.utils.Progbar(target=n_batches)
    for idx, batch in enumerate(dataloader):
        train_loss = train_step(batch)
        pbar.update(idx, values=[("loss",train_loss)])
    val_loss = validate()
    pbar.update(n_batches, values=[('val_loss', val_loss)])


# In[ ]:




