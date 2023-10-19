#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input,Lambda,Dense,Flatten,Conv2D
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[11]:


image_size=[224,224]
train_path='Train'
test_path='Test'


# In[12]:


vgg19=VGG19(input_shape=image_size+[3],weights='imagenet',include_top=False)


# In[13]:


for layer in vgg19.layers:
    layer.trainable=False


# In[18]:


vgg19.summary()


# In[16]:


folders=glob('Train/*')


# In[17]:


folders


# In[19]:


x=Flatten()(vgg19.output)


# In[20]:


prediction=Dense(len(folders),activation='softmax')(x)


# In[21]:


model=Model(inputs=vgg19.input,outputs=prediction)
model.summary()


# In[35]:


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[36]:


train_datagen=ImageDataGenerator(
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen=ImageDataGenerator(
    rescale=1/255
)


# In[37]:


training_set=train_datagen.flow_from_directory('Train',target_size=(224,224),batch_size=32,class_mode='categorical')


# In[38]:


training_set


# In[39]:


test_set=test_datagen.flow_from_directory('Test',target_size=(224,224),batch_size=32,class_mode='categorical')


# In[40]:


test_set


# In[42]:


r=model.fit(
training_set,validation_data=test_set,
epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)


# In[47]:


plt.plot(r.history['loss'],label='train_loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'],label='train_acc')
plt.plot(r.history['val_accuracy'],label='val_acc')
plt.legend()
plt.show()


# In[48]:


from keras.models import load_model
model.save('model_vgg19.h5')


# In[49]:


y_pred=model.predict(test_set)


# In[50]:


y_pred


# In[51]:


y_pred=np.argmax(y_pred,axis=1)


# In[52]:


y_pred


# In[156]:


img=image.load_img('Train/Parasite/C133P94ThinF_IMG_20151004_155721_cell_113.png',target_size=(224,224))


# In[157]:


y=image.img_to_array(img)


# In[158]:


y=y/255
    


# In[159]:


y


# In[160]:


y=np.expand_dims(y,axis=0)
img_data=preprocess_input(y)
img_data.shape


# In[161]:


model.predict(img_data)


# In[162]:


a=np.argmax(model.predict(img_data),axis=1)
a


# In[163]:


if(a==1):
    print("uninfected")
if(a==0):
    print("infected")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




