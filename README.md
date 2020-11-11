This repo shows how to create on-demand random colour augmentations by convolving the image with random conv2d filters.

# rand_conv

The filter weights are initialised from a normal distribution with standard deviation of 1 / sqrt(C_in) * kernel_size

```
    std_normal = 1 / (np.sqrt(3) * kernel_size)

    m.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros_like(m.weight), 
                                               std=torch.ones_like(m.weight)*std_normal))
```



# How to run 

```
python rand_conv.py
```
