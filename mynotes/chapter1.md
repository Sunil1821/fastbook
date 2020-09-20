# Chapter 1 : introduction

#### Code to create your first image recognition model using resnet34 and fastai. 

```
#hide
!pip3 install -Uqq fastbook
import fastbook
fastbook.setup_book()

#hide
from fastbook import *
from fastai.vision.all import *

print(URLs.PETS)
path = untar_data(URLs.PETS)/'images'
print(f"Images downloaded at : {path}")


def is_cat(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

# seed is used so that everytime we get the same validation dataset and training dataset 


learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
# fine tune will run the training for the given number of epochs on your dataset

```

#### Testing our model: 

```
img = PILImage.create(image_cat())
img.to_thumb(192)

is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")

bear_img = PILImage.create(image_bear())
bear_img.to_thumb(192)

is_cat,_,probs = learn.predict(bear_img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```

[Colab](https://colab.research.google.com/drive/1IpGVEVxlkFMVmJAyIu1h-x_-NQPHn4kH?authuser=1#scrollTo=XK8mBNMbGb-1)

##### note:
no need to copy the tensors to gpu it will all be abstracted by the fast ai library. 
