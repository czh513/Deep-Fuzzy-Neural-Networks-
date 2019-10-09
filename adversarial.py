#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


# In[3]:


import torch
from torchvision import datasets, transforms


# In[4]:


torch_model = torch.load('output/cnn-mnist-relog-maxout-sigmoid-out.pkl')
torch_model = lambda x: torch_model(x)[0] # to standard format


# In[5]:


sess = tf.Session()


# In[6]:


x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))


# In[7]:


tf_model_fn = convert_pytorch_model_to_tf(torch_model, out_dims=10)


# In[8]:


cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')


# In[9]:


# Create an FGSM attack
fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
fgsm_params = {'eps': 0.3,
             'clip_min': 0.,
             'clip_max': 1.}
adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
adv_preds_op = tf_model_fn(adv_x_op)


# In[10]:


BATCH_SIZE = 128
test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
  batch_size=BATCH_SIZE)


# In[ ]:


# Run an evaluation of our model against fgsm
total = 0
correct = 0
for xs, ys in test_loader:
    adv_preds = sess.run(adv_preds_op, feed_dict={x_op: xs})
    correct += (np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy()).sum()
    total += test_loader.batch_size


# In[ ]:





# In[ ]:




