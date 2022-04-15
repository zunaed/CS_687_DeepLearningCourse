#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


#4a data generation 
N = 250 ; 
Uh=20;
Ul= -1; 

x1 =np.concatenate([np.random.uniform(Ul, Uh, N), np.random.uniform(-Uh, Ul, N)])
y1 =np.concatenate([np.random.uniform(-Uh, Ul, N), np.random.uniform(Ul, Uh, N)])

x2 =np.concatenate([np.random.uniform(Ul, Uh, N), np.random.uniform(-Uh, Ul, N)])
y2 =np.concatenate([np.random.uniform(Ul, Uh, N), np.random.uniform(-Uh, Ul, N)])

plt.scatter(x1, y1,marker='+', c='blue', label='X-class')
plt.scatter(x2, y2,marker='o', c='red',edgecolors ='none', label='o-class')
#plt.show()
plt.legend(["X-class", "O-class"])
plt.grid(True)


# In[3]:


#4b trainning data establishment (x_train) 
x = np.concatenate((x1, x2), axis=None) 
y = np.concatenate((y1, y2), axis=None) 
x = np.array(x) 
y = np.array(y) 
xy =np.vstack((x,y))
x_train= xy.transpose()
print(x_train)


# In[4]:


#4b trainning data establishment (y_train) 
y_x = [1, 0]
y_o =[0, 1]
y_train= np.array([y_x,y_o])
y_train=np.repeat(y_train, [2*N, 2*N], axis=0)
print(y_train)


# In[12]:


#4c Model Setup
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
print(model.summary())


# In[6]:


#4d Model Training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy'],
)
# fit the keras model on the dataset
history = model.fit(x_train, y_train,validation_split=0.1, epochs=200, batch_size=10, verbose= 1)


# In[7]:


#4e Model Evaluation 
Nt = 75 ; # 150/2 each quadrant 
Uh=20;
Ul= -1; 

x1t =np.concatenate([np.random.uniform(Ul, Uh, Nt), np.random.uniform(-Uh, Ul, Nt)])
y1t =np.concatenate([np.random.uniform(-Uh, Ul, Nt), np.random.uniform(Ul, Uh, Nt)])

x2t =np.concatenate([np.random.uniform(Ul, Uh, Nt), np.random.uniform(-Uh, Ul, Nt)])
y2t =np.concatenate([np.random.uniform(Ul, Uh, Nt), np.random.uniform(-Uh, Ul, Nt)])

xt = np.concatenate((x1t, x2t), axis=None) 
yt = np.concatenate((y1t, y2t), axis=None)  
print(xt) 
print(yt)

xt = np.array(xt) 
yt = np.array(yt) 
print(x)
print(y)
xyt =np.vstack((xt,yt))
x_test= xyt.transpose()
print(x_test)


# In[8]:


#4e Model Evaluation 
y_xt = [1, 0]
y_ot =[0, 1]
y_test= np.array([y_xt,y_ot])
y_test=np.repeat(y_test, [2*Nt, 2*Nt], axis=0)
print(y_test)


# In[9]:


#4e Model Evaluation 
_, score = model.evaluate(x_test, y_test, verbose= 0)

print("Accuracy :", score)


# In[10]:


figure, axes = plt.subplots(nrows=2, ncols=1, figsize = (10, 8))

plt.subplot(211)
plt.title("Loss")
plt.plot(history.history['loss'], label = 'train')
# For validation loss (split)
plt.plot(history.history['val_loss'], label = 'validation')
plt.legend()

plt.subplot(212)
plt.title("Accuracy")
plt.plot(history.history['accuracy'], label = 'train')
# For validation accuracy (split)
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.legend()

figure.tight_layout(pad=3.0)
plt.show()


# In[11]:


# Answer to the question no 4(h)

# Making the labels from two dimension to one dimension
# For (1, 0) = 1 and for (0, 1) = 0
def oneD_from_twoD(y):
  y_oned = []
  for i in range(len(y)):
    if y[i, 0] == 1:
      y_oned.append(1)
    else: y_oned.append(0)
  return(np.asarray(y_oned))

y_test_oned = oneD_from_twoD(y_test)
y_train_oned = oneD_from_twoD(y_train)

# Setting up he model for one dimensional output
model_2 = Sequential()
model_2.add(Dense(8, input_dim=2, activation='sigmoid'))
model_2.add(Dense(1, activation='sigmoid'))

print(model_2.summary())
model_2.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy'],
)

# Training your model
model_2.fit(x_train, y_train_oned, batch_size= 10, epochs= 200, verbose= 1)

# Evaluating model
_, score = model_2.evaluate(x_test,y_test_oned, verbose= 0)

print("Accuracy :", score)

# Plotting decision boundary
def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = x1.min() - 1, y1.max() + 1
    ymin, ymax = x1.min() - 1, y1.max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots(figsize = (10, 8))
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y.ravel(), cmap=cmap, lw=0)

    return fig, ax

plot_decision_boundary(x_test, y_test_oned, model_2, cmap = 'RdBu')


# In[ ]:




