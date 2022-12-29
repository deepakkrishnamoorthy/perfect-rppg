import matplotlib.pyplot as plt
import wandb
import numpy as np

# Initialize a new run
wandb.init(project="visualize-tests", name="images1")

# Generate an image
im = plt.imread(r'C:\Users\Deivid\Desktop\delete\icon.png')

# Log the image
wandb.log({"img": [wandb.Image(im, caption="Cafe")]})

wandb.finish()# -*- coding: utf-8 -*-

x = np.array([0.1,0.2,0.3,0.4,np.nan,np.nan,np.nan,0.5,0.6,0.7,0.8,0.9,1])
y1 = np.array([1,2,3,4,np.nan,np.nan,np.nan,5,6,7,8,9,10])
fig, ax = plt.subplots()
ax.plot(x,y1),
ax.legend(['y1'])
#wandb.log({f'train_pred{epoch}': fig})#wandb.log({f"train_pred/Epoc {epoch}": fig})   

