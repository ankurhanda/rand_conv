import torch 
from torch import nn
import cv2

import matplotlib.pyplot as plt
import numpy as np


'''
read the image here
'''
# img_path = 'images/glasses.jpg'
img_path = 'images/20.png'

# read the images as byte i.e. values are [0, 255]
image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
image_float = np.copy(image).astype('float')/255.0

# create the random_conv filter 
kernel_size = 5
m = nn.Conv2d(3, 3, kernel_size, stride=1, padding=kernel_size//2, bias=False).cuda()

# prepare the data for pytorch tensor 
input_im = torch.from_numpy(image).cuda()
input_im = torch.transpose(input_im, 0, 2)
input_im = input_im.unsqueeze(0)
input_im = input_im.float() / 255.0


fig, ax = plt.subplots(1)
im1 = ax.imshow(image_float,extent=(0, 1, 1, 0))
ax.axis('tight')
ax.axis('off')

# Set whitespace to 0
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# loop for random conv filter 
for i in range(0, 1000):

    std_normal = 1 / (np.sqrt(3) * kernel_size)

    m.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(m.weight), 
                                               std=torch.ones_like(m.weight)*std_normal))
    
    out_im = m(input_im)

    observed = out_im[0]
    observed = torch.transpose(observed, 0, 2)
    observed = observed.cpu().detach().numpy()

    alpha = 0.7 
    observed = alpha * image_float + (1-alpha)*observed
    observed = np.clip(observed, 0.0, 1.0)
    print('max, min = ', np.amax(observed), np.amin(observed))

    im1.set_data(observed)
    fig.show()
    plt.pause(1.0)

plt.show()


