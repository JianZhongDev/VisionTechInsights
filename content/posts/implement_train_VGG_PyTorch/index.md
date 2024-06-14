---
author: "Jian Zhong"
title: "Building and Training VGG with PyTorch: A Step-by-Step Guide"
date: "2024-05-13"
description: "A comprehensive guide on building and training VGG with PyTorch."
tags: ["computer vision", "machine learning"]
categories: ["computer vision", "modeling"]
series: ["computer vision"]
aliases: ["build-train-VGG-PyTorch-guide"]
cover:
   image: images/implement_train_VGG_PyTorch/VGGPyTorch_CoverImage.png
   caption: "[cover image] Architecture of VGG Model (image credit: Jian Zhong)"
ShowToc: true
TocOpen: false
math: true
ShowBreadCrumbs: true
---

The VGG (Visual Geometry Group) model is a type of convolutional neural network (CNN) outlined in the paper [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556v6). It's known for its use of small convolution filters and deep layers, which helped it achieve top-notch performance in tasks like image classification. By stacking multiple layers with small kernel sizes, VGG can capture a wide range of features from input images. Plus, adding more rectification layers makes its decision-making process sharper and more accurate. The paper also introduced 1x1 convolutional layers to enhance nonlinearity without affecting the receptive view. For training, VGG follows the traditional supervised learning approach where input images and ground truth labels are provided.

VGG's architecture has significantly shaped the field of neural networks, serving as a foundation and benchmark for many subsequent models in computer vision.

In this blog post, we'll guide you through implementing and training the VGG architecture using PyTorch, step by step. You can find the complete code for defining and training the VGG model on my [GitHub repository](https://github.com/JianZhongDev/VGGPyTorch) (URL: https://github.com/JianZhongDev/VGGPyTorch ).


## VGG Architecture and Implementation

As you can see in the **cover image** of this post, the VGG model is made up of multiple layers of convolution followed by max-pooling, and it ends with a few fully connected layers. The output from these layers is then fed into a softmax layer to give a normalized confidence score for each image category.

The key features of the VGG network are these stacked convolutional layers and fully connected layers. We will start with these stacked layers in our implementation.

### Stacked Convolutional Layers 

To start, we'll create the stacked convolutional layer as PyTorch `nn.Module`, like this:

```Python {linenos=true}
# stacked 2D convolutional layer
class VGGStacked2DConv(nn.Module):
    def __init__(
            self,
            layer_descriptors = [],
        ):
        assert(isinstance(layer_descriptors, list))
        super().__init__()

        self.network = nn.Identity()

        # create list of stacked layers
        stacked_layers = []

        # iterater through each descriptor for the layers and create corresponding layers
        prev_out_channels = 1
        for i_descrip in range(len(layer_descriptors)):
            cur_descriptor = layer_descriptors[i_descrip]

            # the descriptor needs to be dict
            if not isinstance(cur_descriptor, dict):
                continue
            
            # get input or default values 
            nof_layers = cur_descriptor.get("nof_layers", 1)
            in_channels = cur_descriptor.get("in_channels", prev_out_channels)
            out_channels = cur_descriptor.get("out_channels", 1)
            kernel_size = cur_descriptor.get("kernel_size", 3)
            stride = cur_descriptor.get("stride", 1)
            padding = cur_descriptor.get("padding", 1)
            bias = cur_descriptor.get("bias", True)
            padding_mode = cur_descriptor.get("padding_mode", "zeros")
            activation = cur_descriptor.get("activation", nn.ReLU)
            
            # create layers
            cur_in_channels = in_channels
            for _ in range(nof_layers):
                stacked_layers.append(
                    nn.Conv2d(
                        in_channels = cur_in_channels,
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias = bias,
                        padding_mode = padding_mode,
                    )
                )
                stacked_layers.append(
                    activation()
                )
                cur_in_channels = out_channels
            prev_out_channels = out_channels
            
        # convert list of layers to sequential layers
        if len(stacked_layers) > 0:
            self.network = nn.Sequential(*stacked_layers)

    def forward(self, x):
        y = self.network(x)
        return y

```

The stacked convolutional layer takes in a list of descriptor dictionaries, each detailing the setup for a repeated convolutional layer followed by an activation. It reads these configurations and builds the stacked convolutional layers accordingly. If certain configuration parameters are not specified, the code fills in default values.

### Stacked Fully-Connected and Dropout Layers

VGG uses dropout regularizations in their fully connected layers. Adding the dropout regularization within PyTorch is straightforward: we just need to insert dropout layers after each hidden layer inside the stacked fully connected layer. (NOTE: Section 4.2 of the [AlexNet paper](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) provides valuable insights into dropout layers. It's definitely worth a read.)

We can define the stacked fully connected layer in a similar manner as the stacked convolutional layers:

```Python {linenos=true}
# stacked linear layers
class VGGStackedLinear(nn.Module):
    def __init__(
            self,
            layer_descriptors = [],
    ):
        assert(isinstance(layer_descriptors, list))
        super().__init__()

        self.network = nn.Identity()

        # create list of stacked layers
        stacked_layers = []
        
        # iterater through each descriptor for the layers and create corresponding layers
        prev_out_features = 1
        for i_descrip in range(len(layer_descriptors)):
            cur_descriptor = layer_descriptors[i_descrip]

            # the descriptor needs to be dict
            if not isinstance(cur_descriptor, dict):
                continue            
            
            nof_layers = cur_descriptor.get("nof_layers", 1)
            in_features = cur_descriptor.get("in_features", prev_out_features)
            out_features = cur_descriptor.get("out_features", 1)
            bias = cur_descriptor.get("bias", True)
            activation = cur_descriptor.get("activation", nn.ReLU)
            dropout_p = cur_descriptor.get("dropout_p", None)

            # create layers
            cur_in_features = in_features
            for _ in range(nof_layers):
                stacked_layers.append(
                    nn.Linear(
                        in_features = cur_in_features,
                        out_features = out_features,
                        bias = bias,
                    )
                )
                if activation is not None:
                    stacked_layers.append(
                        activation()
                    )
                if dropout_p is not None:
                    stacked_layers.append(
                        nn.Dropout(p = dropout_p)
                    )
                cur_in_features = out_features
            
            prev_out_features = out_features

        # convert list of layers to sequential layers
        if len(stacked_layers) > 0:
            self.network = nn.Sequential(*stacked_layers)
    
    def forward(self, x):
        y = self.network(x)
        return y

```

### VGG Model

Now that we've defined the stacked convolutional and fully-connected layers, we can construct the VGG model as follows:

```Python {linenos=true}
# VGG model definition
class VGG(nn.Module):
    def __init__(
        self,
        stacked_conv_descriptors,
        stacked_linear_descriptor,
    ):
        assert(isinstance(stacked_conv_descriptors, list))
        assert(isinstance(stacked_linear_descriptor, list))
        super().__init__()

        self.network = nn.Identity()

        stacked_layers = []

        # add stacked convolutional layers and max pooling layers
        for i_stackconv_descrip in range(len(stacked_conv_descriptors)):
            cur_stacked_conv_descriptor = stacked_conv_descriptors[i_stackconv_descrip]
            if not isinstance(cur_stacked_conv_descriptor, list):
                continue
            stacked_layers.append(
                StackedLayers.VGGStacked2DConv(
                    cur_stacked_conv_descriptor
                )
            )

            # add max pooling layer after stacked convolutional layer
            stacked_layers.append(
                nn.MaxPool2d(
                    kernel_size = 2,
                    stride = 2,
                )
            )

        # flatten convolutional layers 
        stacked_layers.append(
            nn.Flatten()
        )
        
        # add stacked linear layers
        stacked_layers.append(
            StackedLayers.VGGStackedLinear(
                stacked_linear_descriptor
            )
        )

        # add softmax layer at the very end
        stacked_layers.append(
            nn.Softmax(dim = -1)
        )

        # convert list of layers to Sequantial network
        if len(stacked_layers) > 0:
            self.network = nn.Sequential(*stacked_layers)

    def forward(self, x):
        y = self.network(x)
        return y

```

The VGG model takes in a stacked convolutional layer descriptor list, and a fully connected layer descriptor. First, it goes through the convolutional layer descriptors, creating stacked convolutional layers for each descriptor and adding a max pooling layer after each set of stacked convolutional layers. Then, it flattens the output from all the convolutional layers and constructs stacked fully connected layers based on the linear layer descriptor. Finally, a Softmax layer is appended at the end of the network.

### Model Generation

Using the model definition provided above, we can create a VGG model by specifying a few layer descriptors. For instance, we can replicate the VGG16 model described in the VGG paper as follows:

```Python {linenos=true}
## Demo creating 16-layer VGG model

input_image_width = 224
input_image_height = 224

model_stacked_conv_list = [
    [ 
        {"nof_layers": 2, "in_channels": 3, "out_channels": 64,}, 
    
    ],
    [ 
        {"nof_layers": 2, "in_ckjhannels": 64, "out_channels": 128,}, 
    ],
    [ 
        {"nof_layers": 2, "in_channels": 128, "out_channels": 256, },
        {"nof_layers": 1, "out_channels": 256, "kernel_size": 1, "padding": 0},
    ],
    [ 
        {"nof_layers": 2, "in_ckjhannels": 256, "out_channels": 512,}, 
        {"nof_layers": 1, "out_channels": 512, "kernel_size": 1, "padding": 0},
    ],
    [ 
        {"nof_layers": 2, "in_ckjhannels": 512, "out_channels": 512,}, 
        {"nof_layers": 1, "out_channels": 512, "kernel_size": 1, "padding": 0},
    ],
]

conv_image_reduce_ratio = 2**len(model_stacked_conv_list)
conv_final_image_width = input_image_width//conv_image_reduce_ratio
conv_final_image_height = input_image_height//conv_image_reduce_ratio

model_stacked_linear = [
    { "nof_layers": 2, 
     "in_features": conv_final_image_width * conv_final_image_height * 512, 
     "out_features": 4096, 
     "dropout_p": 0.5
    },
    { "nof_layers": 1, 
     "out_features": 1000, 
     "activation": None
    }
]

model = VGG.VGG(
    stacked_conv_descriptors = model_stacked_conv_list,
    stacked_linear_descriptor = model_stacked_linear,
    enable_debug = False,
)

print(model)
```

Here's what the printout of the VGG16 model looks like:

{{< details title="click to expand 16-layer VGG model printout">}}
```
VGG(
  (network): Sequential(
    (0): VGGStacked2DConv(
      (network): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
      )
    )
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): VGGStacked2DConv(
      (network): Sequential(
        (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
      )
    )
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): VGGStacked2DConv(
      (network): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (5): ReLU()
      )
    )
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): VGGStacked2DConv(
      (network): Sequential(
        (0): Conv2d(1, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (5): ReLU()
      )
    )
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): VGGStacked2DConv(
      (network): Sequential(
        (0): Conv2d(1, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (5): ReLU()
      )
    )
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Flatten(start_dim=1, end_dim=-1)
    (11): VGGStackedLinear(
      (network): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU()
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )
    (12): Softmax(dim=-1)
  )
)
```
{{< /details >}}
&NewLine;

## Data Processing

In the VGG paper, the only data processing done on the input data is subtracting the RGB value calculated from the training set. To apply this processing, we start by going through the entire training dataset and computing the mean value for each color channel.

```Python {linenos=true}
## calculated the averaged channel values across the entire data set

# train_dataloader is the dataloader iterating through the entire training data set
input_dataloader = train_dataloader
nof_batchs = len(input_dataloader)
avg_ch_vals = [None for _ in range(nof_batchs)]

for i_batch, data in enumerate(input_dataloader):
    inputs, labels = data
    cur_avg_ch = torch.mean(inputs, dim = (-1,-2), keepdim = True)
    avg_ch_vals[i_batch] = cur_avg_ch

avg_ch_vals = torch.cat(avg_ch_vals, dim = 0)
avg_ch_val = torch.mean(avg_ch_vals, dim = 0, keepdim = False)

print("result size = ")
print(avg_ch_val.size())
print("result val = ")
print(repr(avg_ch_val))

```

Using the mean channel value, we can perform the mentioned data processing by defining a background subtraction function and using the `Lambda() ` transform provided by `torchvision` like this:

```Python {linenos=true}
import functools
from torchvision.transforms import v2

# subtract constant value from the image
def subtract_const(src_image, const_val):
    return src_image - const_val

## subtract global mean channel background
train_data_ch_avg = torch.tensor([[[0.4914]],[[0.4822]],[[0.4465]]])
# NOTE: train_data_ch_avg is obtained from the channel mean value calculation code
print(train_data_ch_avg.size())
subtract_ch_avg = functools.partial(subtract_const, const_val = train_data_ch_avg)

subtract_channel_mean_transform = v2.Lambda(subtract_ch_avg)

```

## Data Augmentation

The VGG paper also employed various data augmentation techniques to prevent overfitting. Here's how we implement them:

### Random Horizontal Flip

`torchvision` already includes a built-in transformation for randomly flipping images horizontally. Therefore, we can simply utilize this built-in transformation for horizontal flips.

```Python {linenos=true}
from torchvision.transforms import v2

rand_hflip_transform = v2.RandomHorizontalFlip(0.5)

```

In the VGG paper, they utilized both the original image and its horizontally flipped counterpart to predict classification results. They then averaged these results to obtain the final classification. Consequently, we can implement the validation process as follows:

```Python {linenos=true}
# validate model in one epoch and return the top k-th result 
def validate_one_epoch_topk_aug(
        model, 
        validate_loader, 
        loss_func, 
        transforms, 
        device, 
        top_k = 1
        ):
    
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = len(validate_loader)

    correct_samples = 0
    tot_samples = len(validate_loader.dataset)

    nof_transforms = len(transforms)

    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(validate_loader):
            inputs, labels = data 
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            group_outputs = [None for _ in range(nof_transforms)]
            group_loss = [None for _ in range(nof_transforms)]
            for i_trans in range(nof_transforms):
                cur_transform = transforms[i_trans]
                cur_input = inputs
                if cur_transform is not None:
                    cur_input = cur_transform(inputs)
                cur_output = model(cur_input)
                cur_loss = loss_func(cur_output, labels)
                group_outputs[i_trans] = cur_output
                group_loss[i_trans] = cur_loss
            
            outputs = torch.mean(torch.stack(group_outputs, dim = 0), dim = 0)
            loss = torch.mean(torch.stack(group_loss, dim = 0), dim = 0)
            
            tot_loss += loss.item()
            # NOTE: we will define batch_in_top_k() later
            # NOTE: batch_in_top_k() return a mask array indicate if label in the top k result 
            correct_samples += (batch_in_top_k(outputs, labels, top_k)).type(torch.float).sum().item()

    avg_loss = tot_loss/tot_nof_batch
    correct_rate = correct_samples/tot_samples

    print(f"Validate: top{top_k} Accuracy: {(100*correct_rate):>0.2f}%, Avg loss: {avg_loss:>8f}")

    return (avg_loss, correct_rate)

## demo training validation loop
validate_transforms = [None, torchvision.transforms.functional.hflip]

for i_epoch in range(nof_epochs):
    print(f" ------ Epoch {i_epoch} ------ ")

    train_one_epoch(model, train_dataloader, loss_func, optimizer, device)
    validate_one_epoch_topk_aug(model, validate_dataloader, loss_func, validate_transforms, 
device, top_k)
    
```

### Random Color Shift

In VGG, another augmentation technique involved adjusting the RGB values of training images by a random combination of the principal component analysis (PCA) eigenvectors derived from the RGB values across all pixels of all images in the training set. For a detailed explanation, refer to section *4.1 Data Augmentation* in the [AlexNet paper](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html).

Here's how the random color shift is implemented: 

{{< math.inline >}}
<p>
Before training begins, we go through the entire training set and gather all RGB values from each image. This data is used to create an \(m \times n\) data matrix, where \(n\) represents the number of channels (3 for RGB images) and \(m\) represents the total number of pixels across all images in the training set \(m = \text{number of images} \times \text{image height} \times \text{image width}\). We then calculate the covariance matrix of this data matrix. Next, we conduct principal component analysis (PCA) on the covariance matrix using singular value decomposition (SVD). The resulting \(U\) matrix contains columns representing the PCA eigenvectors, and the \(S\) matrix contains the corresponding eigenvalues.
</p>
{{</ math.inline >}}

```Python {linenos=true}
## PCA for covariance matrix of image channels across all the pixels 

# train_dataloader is the dataloader iterating through the entire training data set
input_dataloader = train_dataloader
nof_batchs = len(input_dataloader)
ch_vecs = [None for _ in range(nof_batchs)]

for i_batch, data in enumerate(input_dataloader):
    inputs, labels = data
    # swap channel and batch axis and flatten the dimension of (batch, image height, image width)
    ch_vecs[i_batch] = torch.flatten(torch.swapaxes(inputs, 0, 1), start_dim = 1, end_dim = -1)

ch_vecs = torch.cat(ch_vecs, dim = -1)
ch_cov = torch.cov(ch_vecs)
ch_vecs = None

U, S, Vh = torch.linalg.svd(ch_cov, full_matrices = True)

## Each column of U is a channel PCA eigenvector
## S contains the corresponding to eigenvectors

print("U:")
print(repr(U))
print("S:")
print(S)
print("Vh:")
print(Vh)
```
Note: In this implementation, all pixels are loaded into computer memory at the same time. For larger datasets, the code for calculating the covariance matrix may need enhancements to compute it without simultaneously loading all data into memory.

During training, we create a randomized linear combination of PCA eigenvectors by adding up the product of each eigenvector with a randomized amplitude. This amplitude is computed by multiplying the corresponding eigenvalue by a random value drawn from a Gaussian distribution with a mean of 0 and a standard deviation of 0.1.

```Python {linenos=true}
import functools
from torchvision.transforms import v2

# image channel radom PCA eigenvec addition agumentation
def random_ch_shift_pca(src_image, pca_eigenvecs, pca_eigenvals, random_paras = None):
    norm_meam = 0
    norm_std = 0.1
    if isinstance(random_paras, dict):
        norm_meam = random_paras.get("mean", norm_meam)
        norm_std = random_paras.get("std", norm_std)
    
    nof_dims = len(src_image.size())
    nof_channels = src_image.size(0)

    assert(pca_eigenvecs.size(0) == nof_channels)
    assert(len(pca_eigenvals.size()) == 1)
    assert(pca_eigenvals.size(0) == pca_eigenvecs.size(1))

    norm_means = 0 * torch.ones(pca_eigenvals.size())
    norm_stds = 0.1 * torch.ones(pca_eigenvals.size())

    alphas = torch.normal(norm_means, norm_stds)
    scale_factors = (alphas * pca_eigenvals).view((-1,1))

    ch_offset = torch.matmul(pca_eigenvecs, scale_factors)

    ch_offset = ch_offset.view((nof_channels,) + (1,) * (nof_dims - 1))

    dst_image = src_image + ch_offset

    return dst_image


## random channel shifts
# NOTE: trainset_pca_eigenvecs and trainset_pca_eigenvals are the U and S matrix obtained from above mentioned PCA analysis
trainset_pca_eigenvecs = torch.tensor([[-0.5580,  0.7063,  0.4356], [-0.5775,  0.0464, -0.8151], [-0.5960, -0.7063,  0.3820]])
print(trainset_pca_eigenvecs.size())
trainset_pca_eigenvals = torch.tensor([0.1719, 0.0139, 0.0029])
print(trainset_pca_eigenvals.size())

random_ch_shift = functools.partial(random_ch_shift_pca, 
                                    pca_eigenvecs = trainset_pca_eigenvecs,
                                    pca_eigenvals = trainset_pca_eigenvals,
                                    random_paras = {"mean": 0, "std": 0.1},
                                   )

random_ch_shift_transform =  v2.Lambda(random_ch_shift)
```

### Other Data Augmentations

The VGG paper also employed additional augmentation techniques like random translations and random crops. However, since the CIFAR dataset's image size is much smaller (32x32) compared to the ImageNet dataset (256x256), there isn't much flexibility to utilize these techniques effectively.

## Summary of Data Transformations 

In summary, the data transformations for the training set, including preprocessing and all data augmentation techniques, can be implemented as follows:


```Python {linenos=true}
from torchvision.transforms import v2

# NOTE: subtract_ch_avg() is defined in the Data processing section
# NOTE: random_ch_shift() is defined in the Random color shift section

## data transform for training
train_data_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale = True),
    v2.Lambda(subtract_ch_avg),
    v2.RandomHorizontalFlip(0.5),
    v2.Lambda(random_ch_shift),
])
```

For the test/validation set, all we need to do is include the preprocessing step in the data transformations.

```Python {linenos=true}
from torchvision.transforms import v2

# NOTE: subtract_ch_avg() is defined in the Data processing section

## data transform for validation
validate_data_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale = True),
    v2.Lambda(subtract_ch_avg),
])
```

## Training and Validation

### Top k Accuracy (or Error)

In the VGG paper, the main way they measured performance was using the top k error. In my version, I focused on calculating the top k accuracy instead. Top k accuracy shows how often the actual label is among the top k predictions made by the model with the highest confidence. On the other hand, top k error tells us how often the actual label is not included in the top k predictions.

The relationship between top k error and top k accuracy is simply connected by the following formula:

$$
\text{top k error} = 1 − \text{top k accuracy}
$$

A higher top k accuracy and lower top k error indicate better model performance.

During the validation (or test) process, the top k-th accuracy can be estimated by dividing the total number of valiation (or test) samples by the number of samples where the label is in the top k predictions.

$$
\text{top k accuracy} = \frac{\text{number of samples (label is in top k predictions)}}{\text{total number of samples}}
$$

Therefore, top k accuracy can be calculated using the following code:

```Python {linenos=true}
# Evaluate if label is within top k prediction result for one batch of data
def batch_in_top_k(outputs, labels, top_k = 1):
    sorted_outputs, sorted_idxs = torch.sort(outputs, dim = -1, descending = True)
    in_top_k = torch.full_like(labels, False)
    for cur_idx in range(top_k):
        in_top_k = torch.logical_or(sorted_idxs[:,cur_idx] == labels, in_top_k)
    return in_top_k   

```

In each batch, we organize the softmax layer results, which represent the confidences for each predicted category, in descending order. Then, we check if the ground truth label is among the top k predictions. This check result is stored in a boolean mask array, where 'true' indicates the label is in the top k predictions, and 'false' indicates it's not. This boolean mask array holds the results for all samples within the batch. To find the total number of samples where the label is among the top k predictions, we simply sum the mask arrays from all batches.

### Loss Function, Regularization, and Optimizer

VGG employs multinomial logistic regression as its loss function. For optimization, it utilizes mini-batch gradient descent with momentum and weight decay. In PyTorch, these can be implemented as follows:

```Python {linenos=true}
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1E-2, momentum = 0.9, weight_decay= 5E-4)
```

Additionally, dropout regularization has been incorporated into the model as another form of regularization as mentioned earlier in this post.

### Learning Rate Adjustment

In the VGG paper, the authors initially train with a learning rate of 1E-2. Then, they reduce the learning rate by a factor of 10 when the validation set accuracy plateaus. This can be implemented using the `ReduceLROnPlateau()` function provided by PyTorch, like this:

```Python {linenos=true}
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    mode = "max",
    factor = 0.1,
    patience = 10,
    threshold = 1E-3,
    min_lr = 0,
)

# Demo training and validation loop
for i_epoch in range(nof_epochs):
    print(f" ------ Epoch {i_epoch} ------ ")

    cur_lr = optimizer.param_groups[0]['lr'];
    
    print(f"current lr = {cur_lr}")

    cur_train_loss = train_one_epoch(model, 
                                     train_dataloader,  
                                     loss_func, 
                                     optimizer, 
                                     device)
    cur_validate_loss, cur_validate_accuracy = validate_one_epoch_topk_aug(model, 
                                                                           validate_dataloader, 
                                                                           loss_func, 
                                                                           validate_transforms, 
                                                                           device, 
                                                                           top_k)

    scheduler.step(cur_validate_accuracy)
    
    print("\n")

```

NOTE: The description of the `ReduceLROnPlateau()` function in the PyTorch documentation can be confusing. I found that reading the source code of the `ReduceLROnPlateau()` definition provides clearer understanding.

### Training Deep Models

Optimizing deep models from scratch with completely random initialization can be very challenging for the optimizer. It often leads to the learning process getting stuck for long periods.

To tackle this issue, the VGG authors first train a shallow model. Then, they use the learned parameters from this shallow model to initialize deeper ones.

Transferring learned parameters between models in PyTorch is straightforward. It involves copying the learnable parameters `state_dict` (i.e. weights and biases) from corresponding layers between the two models. If you're using the VGG model definition from this blog post, the example code looks like this:

```Python {linenos=true}
## demo transfer model parameters

input_image_width = 32
input_image_height = 32

# create model 1
model1_stacked_conv_list = [
    [ 
        {"nof_layers": 1, "in_channels": 3, "out_channels": 64,}, 
    
    ],
    [ 
        {"nof_layers": 1, "in_ckjhannels": 64, "out_channels": 128,}, 
    ],
]
conv_image_reduce_ratio = 2**len(model1_stacked_conv_list)
conv_final_image_width = input_image_width//conv_image_reduce_ratio
conv_final_image_height = input_image_height//conv_image_reduce_ratio
model1_stacked_linear = [
    { "nof_layers": 1, 
     "in_features": conv_final_image_width * conv_final_image_height * 512, 
     "out_features": 512, 
     "dropout_p": 0.5
    },
    { "nof_layers": 1, 
     "out_features": 10, 
     "activation": None
    }
]
model1 = VGG.VGG(
    stacked_conv_descriptors = model1_stacked_conv_list,
    stacked_linear_descriptor = model1_stacked_linear,
    enable_debug = False,
)

# create model 2
model2_stacked_conv_list = [
    [ 
        {"nof_layers": 1, "in_channels": 3, "out_channels": 64,}, 
    
    ],
    [ 
        {"nof_layers": 2, "in_ckjhannels": 64, "out_channels": 128,}, 
    ],
]
conv_image_reduce_ratio = 2**len(model2_stacked_conv_list)
conv_final_image_width = input_image_width//conv_image_reduce_ratio
conv_final_image_height = input_image_height//conv_image_reduce_ratio
model2_stacked_linear = [
    { "nof_layers": 1, 
     "in_features": conv_final_image_width * conv_final_image_height * 512, 
     "out_features": 512, 
     "dropout_p": 0.5
    },
    { "nof_layers": 1, 
     "out_features": 10, 
     "activation": None
    }
]
model2 = VGG.VGG(
    stacked_conv_descriptors = model2_stacked_conv_list,
    stacked_linear_descriptor = model2_stacked_linear,
    enable_debug = False,
)

# transfer parameter of 1st convoluation layer from model 1 to model 2
model2.network[0].network[0].load_state_dict(model1.network[0].network[0].state_dict())
```
NOTE: We've organized the sequential layers of the VGG model and stacked them within the "network" attribute of the object. This means we can access each specific layer inside the network by indexing the "network" attribute.

## Results

Given the large size of the ImageNet dataset and the extensive time required for training, we'll opt for a smaller dataset, [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), to demonstrate training and validation more quickly.


I've examined several models based on the VGG architecture, and I've listed some of them (model I, II, and III) below:

{{< html >}}
<table>
    <style>
    table {
      font-family: arial, sans-serif;
      border-collapse: collapse;
      width: 100%;
    }
    td, th {
      border: 1px solid #dddddd;
      text-align: center;
      padding: 8px;
    }
    </style>
    <thead>
        <tr>
            <td colspan=3>Model Configuration</td>
        </tr>
        <tr>
            <th>I</th>
            <th>II</th>
            <th>III</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>conv3-128</td>
            <td>conv3-128</td>
            <td>conv3-128</td>
        </tr>
        <tr>
            <td colspan=3>maxpool</td>
        </tr>
        <tr>
            <td rowspan=2>conv3-256</td>
            <td>conv3-256</td>
            <td>conv3-256</td>
        </tr>
        <tr>
            <td>conv3-256</td>
            <td>conv3-256</td>
        </tr>
        <tr>
            <td colspan=3>maxpool</td>
        </tr>
        <tr>
            <td rowspan=6>conv3-512</td>
            <td rowspan=3>conv3-512</td>
            <td>conv3-512</td>
        </tr>
        <tr>
            <td>conv3-512</td>
        </tr>
        <tr>
            <td>conv3-512</td>
        </tr>
        <tr>
            <td rowspan=3>conv3-512</td>
            <td>conv3-512</td>
        </tr>
        <tr>
            <td>conv3-512</td>
        </tr>
        <tr>
            <td>conv3-512</td>
        </tr>
        <tr>
            <td colspan=3>maxpool</td>
        </tr>
        <tr>
            <td rowspan=2>FC-1024</td>
            <td>FC-1024</td>
            <td>FC-1024</td>
        </tr>
        <tr>
            <td>FC-1024</td>
            <td>FC-1024</td>
        </tr>
        <tr>
            <td>FC-10</td>
            <td>FC-10</td>
            <td>FC-10</td>
        </tr>
        <tr>
            <td colspan=3>soft-max</td>
        </tr>

    </tbody>
</table>
{{< /html >}}


After training these model variations, I computed the top 1 to top 5 accuracies using the CIFAR10 test dataset. Here's a summary of the results:

{{< html >}}
<table>
    <style>
    table {
      font-family: arial, sans-serif;
      border-collapse: collapse;
      width: 100%;
    }
    td, th {
      border: 1px solid #dddddd;
      text-align: center;
      padding: 8px;
    }
    </style>
    <thead>
        <tr>
            <th>Model config.</th>
            <th>top-1 accuarcy(%)</th>
            <th>top-2 accuarcy(%)</th>
            <th>top-3 accuarcy(%)</th>
            <th>top-4 accuarcy(%)</th>
            <th>top-5 accuarcy(%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>I</td>
            <td>82.45</td>
            <td>92.74</td>
            <td>96.23</td>
            <td>97.82</td>
            <td>98.87</td>
        </tr>
        <tr>
            <td>II</td>
            <td>84.88</td>
            <td>93.95</td>
            <td>96.91</td>
            <td>98.23</td>
            <td>98.99</td>
        </tr>
        <tr>
            <td>III</td>
            <td>86.93</td>
            <td>94.39</td>
            <td>96.83</td>
            <td>98.15</td>
            <td>98.90</td>
        </tr>
    </tbody>
</table>
{{< /html >}}

We can observe that the accuracy tends to improve as the depth of the models increases.

## Conclusion

In this blog post, we've covered the implementation, training, and evaluation of the VGG network in a step-by-step manner. The VGG model showcases the effectiveness of deep neural networks in tackling image classification tasks. Moreover, their methods for data augmentation, regularization, and training provide valuable insights and lessons for training deep neural networks.

## Reference

[1] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for Large-Scale image recognition. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1409.1556

[2] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. Neural Information Processing Systems, 25, 1097–1105. https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

[3] Krizhevsky, A., Nair, V. and Hinton, G. (2014) The CIFAR-10 Dataset. https://www.cs.toronto.edu/~kriz/cifar.html

## Citation

If you found this article helpful, please cite it as:
> Zhong, Jian (May 2024). Building and Training VGG with PyTorch: A Step-by-Step Guide. Vision Tech Insights. https://jianzhongdev.github.io/VisionTechInsights/posts/implement_train_vgg_pytorch/.

Or

```html
@article{zhong2024buildtrainVGGPyTorch,
  title   = "Building and Training VGG with PyTorch: A Step-by-Step Guide",
  author  = "Zhong, Jian",
  journal = "jianzhongdev.github.io",
  year    = "2024",
  month   = "May",
  url     = "https://jianzhongdev.github.io/VisionTechInsights/posts/implement_train_vgg_pytorch/"
}
```



