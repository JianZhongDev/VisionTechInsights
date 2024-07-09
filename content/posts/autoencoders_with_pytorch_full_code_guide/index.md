---
author: "Jian Zhong"
title: "Autoencoders with PyTorch: Full Code Guide"
date: "2024-06-23"
description: "A comprehensive guide on building and training autoencoders with PyTorch."
tags: ["computer vision", "machine learning"]
categories: ["computer vision", "modeling"]
series: ["computer vision"]
aliases: ["autoencoders-with-Pytorch-full-code-guide"]
cover:
   image: images/autoencoders_with_pytorch_full_code_guide/AutoencoderCoverImage.png
   caption: "[cover image] Architecture of Autoencoder (image credit: Jian Zhong)"
ShowToc: true
TocOpen: false
math: true
ShowBreadCrumbs: true
---

An autoencoder is a type of artificial neural network that learns to create efficient codings, or representations, of unlabeled data, making it useful for unsupervised learning. Autoencoders can be used for tasks like reducing the number of dimensions in data, extracting important features, and removing noise. They're also important for building semi-supervised learning models and generative models. The concept of autoencoders has inspired many advanced models.

In this blog post, we'll start with a simple introduction to autoencoders. Then, we'll show how to build an autoencoder using a fully-connected neural network. We'll explain what sparsity constraints are and how to add them to neural networks. After that, we'll go over how to build autoencoders with convolutional neural networks. Finally, we'll talk about some common uses for autoencoders.

You can find all the source code and tutorial scripts mentioned in this blog post in my [GitHub repository](https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main) (URL: https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main ).

## Autoencoder Network

### Redundancy of Data Representation 

The key idea behind autoencoders is to reduce redundancy in data representation. Often, data is represented in a way that isn't very efficient, leading to higher dimensions than necessary. This means many parts of the data are redundant. For example, the MNIST dataset contains 28x28 pixel images of handwritten digits from 0 to 9. Ideally, we only need one variable to represent these digits, but the image representation uses 784 (28x28) grayscale values.

Autoencoders work by compressing the features as the neural network processes the data and then reconstructing the original data from this compressed form. This process helps the network learn a more efficient way to represent the input data.

### Typical Structure of an Autoencoder Network

An autoencoder network typically has two parts: an encoder and a decoder. The encoder compresses the input data into a smaller, lower-dimensional form. The decoder then takes this smaller form and reconstructs the original input data. This smaller form, created by the encoder, is often called the latent space or the "bottleneck." The latent space usually has fewer dimensions than the original input data.

{{< figure src="./Images/AutoencoderCoverImage.png" attr="Architecture of autoencoder. (image credit: Jian Zhong)" align=center target="_blank" >}}

## Fully-Connected Autoencoder

Implementing an autoencoder using a fully connected network is straightforward. For the encoder, we use a fully connected network where the number of neurons decreases with each layer. For the decoder, we do the opposite, using a fully connected network where the number of neurons increases with each layer. This creates a "bottleneck" structure in the middle of the network.

Here is a code example demonstrating how to implement the encoder and decoder of a simple autoencoder network using fully-connected neural networks.

```Python {linenos=true}
from .Layers import StackedLayers

## fully connected network with only fully connected layers 
class SimpleFCNetwork(nn.Module):
    def __init__(
        self,
        layer_descriptors = [],
    ):
        assert isinstance(layer_descriptors, list)
        super().__init__()
        self.network = StackedLayers.VGGStackedLinear(layer_descriptors)
    
    def forward(self, x):
        y = self.network(x)
        return y

## create models using the above Module 
nof_features = 28 * 28
code_dim = 64

## create encoder model
encoder_layer_descriptors = [
    {"nof_layers": 1, "in_features": nof_features, "out_features": code_dim, "activation": torch.nn.LeakyReLU},
]

encoder = SimpleFCNetwork(
    layer_descriptors = encoder_layer_descriptors
)

print("Encoder:")
print(encoder)

print("\n")

## create decoder model
decoder_layer_descriptors = [
    {"nof_layers": 1, "in_features": code_dim, "out_features": nof_features, "activation": torch.nn.LeakyReLU},
]

decoder = SimpleFCNetwork(
    layer_descriptors = decoder_layer_descriptors
)

print("Decoder:")
print(decoder)
```

The VGGStackedLinear module creates several fully-connected networks based on the input layer descriptors. For a detailed explanation, please refer to my blog post on [building and training VGG network with PyTorch](../implement_train_VGG_PyTorch/index.md).

Here's how the architecture of the encoder and decoder defined above looks:

{{< details title="click to expand simple fully-connected autoencoder printout">}}
```
Encoder:
SimpleFCNetwork(
  (network): VGGStackedLinear(
    (network): Sequential(
      (0): Linear(in_features=784, out_features=64, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
    )
  )
)


Decoder:
SimpleFCNetwork(
  (network): VGGStackedLinear(
    (network): Sequential(
      (0): Linear(in_features=64, out_features=784, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
    )
  )
)

```
{{< /details >}}
&NewLine;

After training the fully-connected network, here are the results for an example data input/output, the latent representation of data in a batch of 512 samples, and the learned feature dictionary:

{{< figure src="./Images/SimpleFCAutoEncoderResult.png" attr="Training results of a simple fully-connected autoencoder (encoder: 784-64, decoder 64-784). **a,** example data input/output. **b,** latent representation of data in a batch of 512 samples. **c,** the learned (decoder) feature dictionary. (image credit: Jian Zhong)" align=center target="_blank" >}}

Without additional constraints, each sample typically contains numerous non-zero latent features of similar amplitudes, and the learned feature dictionary tends to be highly localized.

For a comprehensive understanding of how the above network was implemented and trained, please refer to the [TrainSimpleFCAutoencoder Jupyter notebook](https://github.com/JianZhongDev/AutoencoderPyTorch/blob/main/TrainSimpleFCAutoencoder.ipynb) in my [GitHub repository](https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main).


## Sparsity and Sparse Autoencoder

In machine learning, sparsity suggests that in many high-dimensional datasets, only a small number of features or variables are meaningful or non-zero for each observation. In an optimal representation space, many features either have zero values or values that are negligible.

In the context of autoencoders, a sparse latent representation of the data is often preferred. This sparse representation can be achieved by incorporating sparse constraints into the network. Adding these constraints helps the autoencoder focus on learning more meaningful features.

### Hard Sparsity in Latent Representation

Implementing hard sparsity in the latent space involves adding a sparsity layer at the end of the encoder network along the feature dimension.
To create a hard sparsity layer, we specify a number k of features to retain in the latent space. During the forward pass, this layer keeps only the top k largest features of the encoded representation for each sample, setting the rest to 0. During backward propagation, the hard sparsity layer only propagates gradients for these top k features.

Here's how the hard sparsity layer is implemented:

```Python {linenos=true}
# hard sparsity function to select the largest k features for each sample in the batch input data
# NOTE: this function works on 1d feature space
class FeatureTopKFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k):
        assert len(x.size()) == 2

        src_data_detach = x.detach() 

        # create mask indicating the top k features for each sample within the feature space
        topk_mask = torch.zeros_like(x, dtype = bool, requires_grad = False)
        _, indices = src_data_detach.topk(k, dim = -1)
        for i_batch in range(x.size(0)):
            topk_mask[i_batch, indices[i_batch,:]] = True

        # save mask for backward propagation
        ctx.save_for_backward(topk_mask)

        # only propagate largest k features of each sample 
        y = torch.zeros_like(x)
        y[topk_mask] = x[topk_mask]

        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        topk_mask = ctx.saved_tensors[0]
        
        # only propagate gradient for largest k features of each sample
        grad_input = torch.zeros_like(grad_output, requires_grad = True)
        grad_input[topk_mask] = grad_output[topk_mask]

        return grad_input, None

# hard sparsity layer 
class TopKSparsity(nn.Module):
    def __init__(self, topk = 1):
        super().__init__()
        self.topk = topk

    def __repr__(self):
        return self.__class__.__name__ + f"(topk = {self.topk})"

    def forward(self, x):
        y = FeatureTopKFunction.apply(x, self.topk)
        return y
```

First, we created our own operation `FeatureTopKFunction` for hard sparsity and defined its functions for both forward and backward passes. During the forward pass, a mask is generated to identify the top k features of each input sample, which is then stored for later use in the backward pass. This mask ensures that only the top k values are kept, while the rest are set to zero for both value and gradient calculations. In the hard sparsity layer, we specify the number k and incorporate the hard sparsity operation into the `forward()` method. 

To implement hard sparsity in an autoencoder, simply add a hard sparsity layer at the end of the encoder network as follows:

```Python {linenos=true}
# fully connected network with sparsity layer
class SimpleSparseFCNetwork(nn.Module):
    def __init__(
        self,
        layer_descriptors = [],
        feature_sparsity_topk = None,
    ):
        assert isinstance(layer_descriptors, list)

        super().__init__()

        self.network = nn.Identity()

        network_layers = []

        # add stacked fully connected layers
        network_layers.append(StackedLayers.VGGStackedLinear(layer_descriptors))

        # add top k sparsity along the feature dimension
        if feature_sparsity_topk is not None :
            network_layers.append(SparseLayers.TopKSparsity(feature_sparsity_topk))

        if len(network_layers) > 0:
            self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        y = self.network(x)
        return y
```

After training the fully-connected network with these hard sparsity constraints, here are the outcomes for a sample data input/output, the latent representations of data in a batch of 512 samples, and the learned feature dictionary:

{{< figure src="./Images/SimpleFCHardSparsityAutoencoderResult.png" attr="Training results of a simple fully-connected autoencoder with hard sparsity (encoder: 784-64-sparsity, decoder 64-784). **a-c,** results of autoencoder trained with top 16 sparsity. **d-f,** results of autoencoder trained with top 5 sparsity. **a,d,** example data input/output. **b,e,** latent representation of data in a batch of 512 samples. **c,f,** the learned (decoder) feature dictionary. (image credit: Jian Zhong)" align=center target="_blank" >}}

From the results above, we observe that increasing the required sparsity with hard constraints reduces the number of non-zero features in the latent space. This encourages the network to learn more global features.

For a detailed understanding of how this network was implemented and trained, please refer to the [TrainSimpleSparseFCAutoencoder Jupyter notebook](https://github.com/JianZhongDev/AutoencoderPyTorch/blob/main/TrainSimpleSparseFCAutoencoder.ipynb) in my [GitHub repository](https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main).


### Soft Sparsity in Latent Representation

We can also encourage sparsity in the encoded features of the latent space by applying a soft constraint. This involves adding an additional penalty term to the loss function. The modified loss function with the sparsity penalty appears as follows:

$$
H_{\theta}(pred,gt) = J_{\theta}(pred,gt) + \lambda \cdot L_{\theta}(code)
$$

Here, {{< math.inline >}} \(\theta, pred, gt\) {{</ math.inline >}} represents the parameters of the autoencoder network, the output prediction of autoencoder, and the ground truth data, respectively. {{< math.inline >}} \(H_{\theta}(pred,gt)\) {{</ math.inline >}}​ is the loss function with sparsity constraints, where {{< math.inline >}} \(J_{\theta}(pred,gt)\) {{</ math.inline >}} is the original loss function, which measures the difference between the network prediction and ground truth. {{< math.inline >}} \(L_{\theta}(pred,gt)\) {{</ math.inline >}}​ denotes the penalty term for enforcing sparsity. The parameter {{< math.inline >}} \(\lambda\) {{</ math.inline >}} controls the strength of this penalty.

The L1 loss of the encoded features is commonly used as a sparsity loss. This loss function is readily available in PyTorch.

Another approach to implementing sparsity loss is through a penalty based on [KL divergence](https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf). The penalty term for this KL divergence-based sparsity can be defined as follows:

$$
L_{\theta} = \frac{1}{s} \sum^{s}_{j=1} KL(\rho||\hat{\rho_j}) 
$$

Here, ​{{< math.inline >}} \(s\) {{</ math.inline >}} represents the number of features in the encoded representation, which corresponds to the dimension of the latent space. ​{{< math.inline >}} \(j\) {{</ math.inline >}} is index for the features in the latent space. {{< math.inline >}} \(KL(\rho||\hat{\rho_j})\) {{</ math.inline >}} is calculated as follows:

$$
KL(\rho||\hat{\rho_j}) = \rho \cdot log(\frac{\rho}{\hat{\rho}_j}) + (1 - \rho) \cdot log(\frac{1-\rho}{1-\hat{\rho}_j}) 
$$

Here,{{< math.inline >}} \(\rho\) {{</ math.inline >}} is a sparsity parameter, typically a small value close to zero that is provided during training. {{< math.inline >}} \(\hat{\rho}_j\) {{</ math.inline >}}​ is computed from the j-th latent features of the samples within the mini-batch as follows:

$$
\hat{\rho_{j}} = \frac{1}{m} \sum^{m}_{i=1} l_i
$$

Here, {{< math.inline >}} \(m\) {{</ math.inline >}} denotes the batch size. {{< math.inline >}} \(j\) {{</ math.inline >}} indexes the features within the latent space. {{< math.inline >}} \(i\) {{</ math.inline >}} indexes the samples within the minibatch. {{< math.inline >}} \(l\) {{</ math.inline >}} represents each individual feature within the latent space.

Note that for the KL divergence expression, the values of {{< math.inline >}} \(\rho\) {{</ math.inline >}} and {{< math.inline >}} \(\hat{\rho}_j\) {{</ math.inline >}}​ must fall within the range {{< math.inline >}} \((0,1)\) {{</ math.inline >}}. This range should be ensured by using suitable activation functions (such as sigmoid) for the output layer of the encoder, or by appropriately normalizing the latent space features before computing the sparsity loss.

Below is the PyTorch code implementation for the KL-divergence based sparsity loss:

```Python {linenos=true}
# Kullback-Leibler divergence formula
def kullback_leibler_divergence(
        rho, rho_hat    
):
    return rho * torch.log(rho/rho_hat) + (1 - rho)*torch.log((1 - rho)/(1 - rho_hat))


# nn.Module of sparsity loss function 
class KullbackLeiblerDivergenceLoss(nn.Module):
    def __init__(self, rho = 0.05):
        assert rho > 0 and rho < 1
        super().__init__()
        self.rho = rho 

    def forward(self, x):
        rho_hat = torch.mean(x, dim = 0)
        kl = torch.mean(kullback_leibler_divergence(self.rho, rho_hat))
        return kl
```

After training a basic fully-connected autoencoder model with soft sparsity constraints, the results are as follows: 

{{< figure src="./Images/SimpleFCSoftSparsityAutoencoderResult.png" attr="Training results of a simple fully-connected autoencoder with soft sparsity (encoder: 784-64, decoder 64-784, KL-divergence soft sparsity loss {{< math.inline >}} \(\rho = 0.05\) {{</ math.inline >}}). **a-c,** results of autoencoder trained with {{< math.inline >}} \(\lambda = 10^{-2}\) {{</ math.inline >}}. **d-f,** results of autoencoder trained with {{< math.inline >}} \(\lambda = 10^{-1}\) {{</ math.inline >}}. **a,d,** example data input/output. **b,e,** latent representation of data in a batch of 512 samples. **c,f,** the learned (decoder) feature dictionary. (image credit: Jian Zhong)" align=center target="_blank" >}}

Increasing the strength of the sparsity penalty decreases the number of non-zero features in the latent space.

For a comprehensive understanding of how this network was implemented and trained, please refer to the [TrainSimpleFCAutoencoderWithSparseLoss Jupyter notebook](https://github.com/JianZhongDev/AutoencoderPyTorch/blob/main/TrainSimpleFCAutoencoderWithSparseLoss.ipynb) in my [GitHub repository](https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main).

### Lifetime (Winner-Takes-All) Sparsity 

Unlike conventional sparsity constraints that aim to increase sparsity within each individual sample, lifetime sparsity enforces sparsity across minibatch samples for each feature. Here's how lifetime sparsity can be implemented:

During training, in the forward propagation phase, for each feature in the latent space, we retain the top k largest values across all minibatch samples and set the remaining values of that feature to zero. During backward propagation, gradients are propagated only for these k non-zero values.

During testing, we disable the lifetime sparsity constraints, allowing the encoder network to output the final representation of the input.
The implementation of lifetime sparsity operations is as follows:

```Python {linenos=true}
# lifetime sparsity functon to select the largest k samples for each feature
# NOTE: this function works on 1d feature space 
class LifetimeTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        assert len(x.size()) == 2

        k = min(k, x.size(0))

        src_data_detach = x.detach()

        # create mask indicating the top k samples for each feature along the batch dimension
        topk_mask = torch.zeros_like(x, dtype = bool, requires_grad = False)
        _, indices = src_data_detach.topk(k, dim = 0) 
        for i_feature in range(x.size(-1)):
            topk_mask[indices[:,i_feature],i_feature] = True

        # save mask indicationg the top k samples for each feature for back propagation
        ctx.save_for_backward(topk_mask)

        # only propagate largest k samples for each feature
        y = torch.zeros_like(x)
        y[topk_mask] = x[topk_mask]

        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        topk_mask = ctx.saved_tensors[0]
        
        # only propagate gradient for largest k samples for each feature
        grad_input = torch.zeros_like(grad_output, requires_grad = True)
        grad_input[topk_mask] = grad_output[topk_mask]

        return grad_input, None
```

In the forward pass, we create a mask that identifies the top k values across the minibatch dimension for each feature in the latent space. This mask is saved for use during the backward pass. During both forward and backward passes, this mask ensures that only the top k values of each feature are retained, while the rest are set to zero.

With these lifetime sparsity operations, we can implement a neural network layer that enforces lifetime sparsity as follows:

```Python {linenos=true}
# lifetime sparsity layer
class LifetimeTopkSparsity(nn.Module):
    def __init__(self, topk = 5):
        super().__init__()
        self.topk = topk

    def __repr__(self):
        return self.__class__.__name__ + f"(topk = {self.topk})"
    
    def forward(self, x):
        y = None
        if self.training:
            # only apply lifetime sparsity during training
            y = LifetimeTopKFunction.apply(x, self.topk)
        else:
            y = x
        return y
```

In the lifetime sparsity layer, we store the k values within the network object. During training, this layer implements lifetime sparsity operations. During testing, the layer simply passes the input directly to the output.

To implement lifetime sparsity in an autoencoder, we add the lifetime sparsity layer at the end of the encoder network as follows:

```Python {linenos=true}
# fully connected network with sparsity layer
class SimpleSparseFCNetwork(nn.Module):
    def __init__(
        self,
        layer_descriptors = [],
        lifetime_sparsity_topk = None,
    ):
        assert isinstance(layer_descriptors, list)

        super().__init__()

        self.network = nn.Identity()

        network_layers = []

        # add stacked fully connected layers
        network_layers.append(StackedLayers.VGGStackedLinear(layer_descriptors))

        # add top k sparsity along the sample(batch) dimension
        if lifetime_sparsity_topk is not None:
            network_layers.append(SparseLayers.LifetimeTopkSparsity(lifetime_sparsity_topk))

        if len(network_layers) > 0:
            self.network = nn.Sequential(*network_layers)


    def forward(self, x):
        y = self.network(x)
        return y
```

After training a simple fully-connected autoencoder model with a lifetime sparsity layer, the results are as follows: 

{{< figure src="./Images/SimpleFCLifeTimeSparsityAutoencoderResult.png" attr="Training results of a simple fully-connected autoencoder with life time sparsity (encoder: 784-64-sparsity, decoder 64-784). **a-c,** results of autoencoder trained with top 25% sparsity. **d-f,** results of autoencoder trained with top 5% sparsity. **a,d,** example data input/output. **b,e,** latent representation of data in a batch of 512 samples. **c,f,** the learned (decoder) feature dictionary. (image credit: Jian Zhong)" align=center target="_blank" >}}

Increasing the strength of the lifetime sparsity constraint reduces the number of non-zero features in the latent space. This encourages the network to learn more global features. 

For detailed insights into how this network was implemented and trained, please refer to the [TrainSimpleSparseFCAutoencoder Jupyter notebook](https://github.com/JianZhongDev/AutoencoderPyTorch/blob/main/TrainSimpleSparseFCAutoencoder.ipynb) in my [GitHub repository](https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main).

## Convolutional Autoencoder 

For image data, the encoder network can also be implemented using a convolutional network, where the feature dimensions decrease as the encoder becomes deeper. Max pooling layers can be added to further reduce feature dimensions and induce sparsity in the encoded features.
Here's an example of a convolutional encoder network:

```Python {linenos=true}
# simple convolutional encoder = stacked convolutional network + maxpooling 
class SimpleCovEncoder(nn.Module):
    def __init__(
            self,
            convlayer_descriptors = [],
            maxpoollayer_descriptor = {},
    ):
        assert(isinstance(convlayer_descriptors, list))
        assert(isinstance(maxpoollayer_descriptor, dict))

        super().__init__()

        self.network = nn.Identity()

        network_layers = []

        # append stacked convolution layer
        network_layers.append(StackedLayers.VGGStacked2DConv(convlayer_descriptors))

        # append maxpooling layer
        network_layers.append(
            nn.MaxPool2d(
                kernel_size = maxpoollayer_descriptor.get("kernel_size", 2),
                stride = maxpoollayer_descriptor.get("stride", 2),
                padding = maxpoollayer_descriptor.get("padding", 0),
                dilation = maxpoollayer_descriptor.get("dilation", 1),
            )
        )

        # flatten output feature space
        network_layers.append(nn.Flatten(start_dim = 1, end_dim = -1))

        if len(network_layers) > 0:
            self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        y = self.network(x)
        return y

## create encoder model
encoder_convlayer_descriptors = [
    {
        "nof_layers": 4,
        "in_channels": 1,
        "out_channels": 8,
        "kernel_size": 6,
        "stride": 1,
        "padding": 0,
        "activation": torch.nn.LeakyReLU
    }
]
encoder_maxpoollayer_descriptor = {
    "kernel_size": 2,
    "stride": 2,
}
encoder = ConvAutoencoder.SimpleCovEncoder(
    encoder_convlayer_descriptors,
    encoder_maxpoollayer_descriptor,
)

print("Encoder:")
print(encoder)
```

The `VGGStacked2DConv` module generates multiple convolutional networks based on the input layer descriptors. For a detailed explanation, please refer to my blog post on [building and training VGG network with PyTorch](../implement_train_VGG_PyTorch/index.md).

Here's a visualization of the architecture of the encoder and decoder described above:

{{< details title="click to expand convolutional encoder printout">}}
```
Encoder:
SimpleCovEncoder(
  (network): Sequential(
    (0): VGGStacked2DConv(
      (network): Sequential(
        (0): Conv2d(1, 8, kernel_size=(6, 6), stride=(1, 1))
        (1): LeakyReLU(negative_slope=0.01)
        (2): Conv2d(8, 8, kernel_size=(6, 6), stride=(1, 1))
        (3): LeakyReLU(negative_slope=0.01)
        (4): Conv2d(8, 8, kernel_size=(6, 6), stride=(1, 1))
        (5): LeakyReLU(negative_slope=0.01)
        (6): Conv2d(8, 8, kernel_size=(6, 6), stride=(1, 1))
        (7): LeakyReLU(negative_slope=0.01)
      )
    )
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Flatten(start_dim=1, end_dim=-1)
  )
)
```
{{< /details >}}
&NewLine;

After training the fully-connected network, here are the results for a sample data input/output, the latent representation of data in a batch of 512 samples, and the learned feature dictionary:

{{< figure src="./Images/SimpleConvAutoEncoderResult.png" attr="Training results of a simple autoencoder with convolutional encoder and fully-connected decoder (encoder: Conv6x6-Conv6x6-Conv6x6-Conv6x6-MaxPool2x2, decoder 128-784). **a,** example data input/output. **b,** latent representation of data in a batch of 512 samples. **c,** the learned (decoder) feature dictionary. (image credit: Jian Zhong)" align=center target="_blank" >}}

For a detailed understanding of how this network was implemented and trained, please see the [TrainSimpleConvAutoencoder Jupyter notebook](https://github.com/JianZhongDev/AutoencoderPyTorch/blob/main/TrainSimpleConvAutoencoder.ipynb) in my [GitHub repository](https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main).

## Training and Validation

During training, the optimal encoding of input data is generally unknown. In an autoencoder network, the encoder and decoder are trained concurrently. The encoder processes input data to generate compressed representations, while during testing, the decoder reconstructs the input from these representations. The objective of training is to minimize the discrepancy between the decoder's output and the original input data. Typically, Mean Squared Error (MSE) loss is selected as the optimization loss function for this purpose.

### Training Dataset

When training an autoencoder with image datasets, both the input data and the ground truth are images. Depending on the application of the autoencoder, the input data and ground truth images may not necessarily be identical.

In this blog post, we will use the MNIST dataset for our demonstration. In PyTorch, the MNIST dataset provides handwritten digit images as input data and the corresponding digits as ground truth. To train the autoencoder with MNIST and potentially apply various transformations to both input and ground truth images, we implement the following dataset class. This class converts conventional supervised learning datasets into datasets suitable for autoencoder training.

```Python {linenos=true}
# convert supervised data to autoencoder data
def supdata_to_autoencoderdata(
        supdata,
        feature_transform = None,
        target_transform = None,
):
    src_feature = supdata[0] #extract feature
    
    # NOTE: the usuer of this function is responsible for necessary data duplication
    feature = src_feature
    if feature_transform: 
        feature = feature_transform(feature)
    
    target = src_feature
    if target_transform:
        target = target_transform(target)

    return feature, target

# dataset class of autoencoder using existing supervised learning dataset
class AutoencoderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            src_supdataset,
            feature_transform = None,
            target_transform = None,
    ):
        self.dataset = src_supdataset
        self.feature_transform = feature_transform
        self.target_transform = target_transform 

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        src_data = self.dataset[idx]
        feature, target = supdata_to_autoencoderdata(
            src_data,
            self.feature_transform,
            self.target_transform,
        )
        return feature, target
```

### Training and Validation Process

The training process for one epoch is implemented as follows:

```Python {linenos=true}
# train encoder and decoder for one epoch
def train_one_epoch(
    encoder_model, 
    decoder_model,
    train_loader,
    data_loss_func,
    optimizer,
    code_loss_rate = 0,
    code_loss_func = None,
    device = None,
):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = 0

    encoder_model.train(True)
    decoder_model.train(True)

    for i_batch, data in enumerate(train_loader):        
        inputs, targets = data

        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()
        
        cur_codes = encoder_model(inputs) # encode input data into codes in latent space
        cur_preds = decoder_model(cur_codes) # reconstruct input image
        
        data_loss = data_loss_func(cur_preds, targets) 

        # loss for contraints in the latent space
        code_loss = 0
        if code_loss_func:
            code_loss = code_loss_func(cur_codes)
        loss = data_loss + code_loss_rate * code_loss
        
        loss.backward()
        optimizer.step()
        
        tot_loss += loss.item()
        tot_nof_batch += 1

        if i_batch % 100 == 0:
            print(f"batch {i_batch} loss: {tot_loss/tot_nof_batch: >8f}")

    avg_loss = tot_loss/tot_nof_batch

    print(f"Train: Avg loss: {avg_loss:>8f}")

    return avg_loss
```

Mean Squared Error (MSE) loss is typically used as the loss function during training. For sparse autoencoder training, where a sparsity penalty needs to be incorporated into the loss function, the train for one epoch function accepts inputs for the sparsity penalty and its weight.

The validation process for one epoch can be implemented as follows:

```Python {linenos=true}
# validate encoder and decoder for one epoch
def validate_one_epoch(
    encoder_model,
    decoder_model,
    validate_loader,
    loss_func,
    device = True,
):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = len(validate_loader)
    tot_samples = len(validate_loader.dataset)

    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(validate_loader):
            inputs, targets = data

            if device:
                inputs = inputs.to(device)
                targets = targets.to(device)

            cur_codes = encoder_model(inputs) # encode input data into codes in latent space
            cur_preds = decoder_model(cur_codes) # reconstruct input image
        
            loss = loss_func(cur_preds, targets)
            tot_loss += loss.item()

    avg_loss = tot_loss/tot_nof_batch

    print(f"Validate: Avg loss: {avg_loss: > 8f}")

    return avg_loss
```

### Tying and Untying Layer Weights

When training a fully-connected network with symmetrical encoder and decoder structures, it is recommended to initially share the same weight matrix between corresponding layers of the encoder and decoder. Later, for fine-tuning, the weight matrices are separated. This operation is referred to as 'tying the weights' when they are shared, and 'untying the weights' when they are separated.

In PyTorch, we can implement the operations to tie and untie the encoder-decoder matrices as follows:

```Python {linenos=true}
# create tied linear layer
class WeightTiedLinear(nn.Module):
    def __init__(
            self, 
            src_linear: nn.Linear,
            tie_to_linear: nn.Linear
        ):
        super().__init__()
        assert src_linear.weight.size() == tie_to_linear.weight.t().size()
        self.tie_to_linear = tie_to_linear
        self.bias = nn.Parameter(src_linear.bias.clone())

    # use tie_to_linear layer weigth for foward propagation
    def forward(self, input):
        return F.linear(input, self.tie_to_linear.weight.t(), self.bias)
    
    # return weight of tied linear layer
    @property 
    def weight(self):
        return self.tie_to_linear.weight.t()

# tie weights for symmetrical fully-connected auto encoder network.
def tie_weight_sym_fc_autoencoder(
        encoder_model: nn.Module,
        decoder_model: nn.Module,
        skip_no_grad_layer = False,
):
    # get all the fully connected layers
    encoder_fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in encoder_model.named_modules() if isinstance(cur_module, nn.Linear)]
    decoder_fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in decoder_model.named_modules() if isinstance(cur_module, nn.Linear)]

    # validate if the autoencoder model are symmetric
    assert len(encoder_fc_layers) == len(decoder_fc_layers)

    # tie weights for corresponding layers
    nof_fc_layers = len(encoder_fc_layers)
    for i_layer in range(nof_fc_layers):
        cur_encoder_layer = encoder_fc_layers[i_layer]
        cur_decoder_layer = decoder_fc_layers[nof_fc_layers - 1 - i_layer]

        # skip freezed (no grad) layers if needed
        if skip_no_grad_layer:
            if not cur_decoder_layer["module"].weight.requires_grad:
                continue
            if not cur_decoder_layer["module"].weight.requires_grad:
                continue

        # create tied linear module
        cur_tied_decoder_layermodule = WeightTiedLinear(cur_decoder_layer["module"], cur_encoder_layer["module"])

        # update the corresponding layers
        cur_decoder_indexing_substrs = cur_decoder_layer["indexing_str"].split('.')
        cur_nof_substrs = len(cur_decoder_indexing_substrs)

        cur_substr_slow_idx = 0
        cur_substr_fast_idx = 0

        # iterative access corresponding layers
        cur_model = decoder_model
        while(cur_substr_fast_idx < cur_nof_substrs):
            if cur_decoder_indexing_substrs[cur_substr_fast_idx].isdigit():
                if cur_substr_fast_idx == cur_nof_substrs - 1:
                    cur_model.get_submodule(".".join(cur_decoder_indexing_substrs[cur_substr_slow_idx:cur_substr_fast_idx]))[int(cur_decoder_indexing_substrs[cur_substr_fast_idx])] = cur_tied_decoder_layermodule
                else:
                    cur_model = cur_model.get_submodule(".".join(cur_decoder_indexing_substrs[cur_substr_slow_idx:cur_substr_fast_idx]))[int(cur_decoder_indexing_substrs[cur_substr_fast_idx])]
                cur_substr_slow_idx = cur_substr_fast_idx + 1
            cur_substr_fast_idx += 1                             

    return encoder_model, decoder_model

# untie weights for fully-connected network
def untie_weight_fc_models(
        model: nn.Module,
):
    # get all fully connected layers
    # fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in model.named_modules() if isinstance(cur_module, WeightTiedLinear)]
    fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in model.named_modules() if type(cur_module).__name__ == "WeightTiedLinear"]

    # untie weights
    nof_fc_layers = len(fc_layers)
    for i_layer in range(nof_fc_layers):
        cur_layer = fc_layers[i_layer]

        # create linear module for each tied linear layer
        cur_untied_module = nn.Linear(
            in_features = cur_layer["module"].weight.size(1),
            out_features = cur_layer["module"].weight.size(0),
            bias = cur_layer["module"].bias is None,
            device = cur_layer["module"].weight.device,
            dtype = cur_layer["module"].weight.dtype,
        )

        # update linear module weight and bias from tied linear module 
        cur_untied_module.weight = nn.Parameter(cur_layer["module"].weight.clone())
        cur_untied_module.bias = nn.Parameter(cur_layer["module"].bias.clone())

        # update the corresponding layers
        cur_indexing_substrs = cur_layer["indexing_str"].split('.')
        cur_nof_substrs = len(cur_indexing_substrs)

        cur_substr_slow_idx = 0
        cur_substr_fast_idx = 0

        # iterative access corresponding layers
        cur_model = model
        while(cur_substr_fast_idx < cur_nof_substrs):
            if cur_indexing_substrs[cur_substr_fast_idx].isdigit():
                if cur_substr_fast_idx == cur_nof_substrs - 1:
                    cur_model.get_submodule(".".join(cur_indexing_substrs[cur_substr_slow_idx:cur_substr_fast_idx]))[int(cur_indexing_substrs[cur_substr_fast_idx])] = cur_untied_module
                else:
                    cur_model = cur_model.get_submodule(".".join(cur_indexing_substrs[cur_substr_slow_idx:cur_substr_fast_idx]))[int(cur_indexing_substrs[cur_substr_fast_idx])]
                cur_substr_slow_idx = cur_substr_fast_idx + 1
            cur_substr_fast_idx += 1 

    return model
```

When tying a decoder layer to an encoder layer, we create a dummy linear layer that uses the weight of the corresponding encoder layer for forward and backward propagation. When untying the decoder layer, we create a new linear layer and update its weight and bias based on the dummy linear layer.

Using these tying and untying functions, we can tie and untie corresponding linear layers in the encoder and decoder as follows:

```Python {linenos=true}
# tie weights of encoder and decoder
tie_weight_sym_fc_autoencoder(encoder, decoder)

# untie weights
untie_weight_fc_models(encoder)
untie_weight_fc_models(decoder)
```

### Training Deep Autoencoder

For deeper autoencoder networks, unsupervised training can be done in a greedy, layer-wise manner. We start by training the first layer of the encoder and the last layer of the decoder using the input and ground truth images. Once these layers are trained, we freeze them (disable their weight updates) and add the second layer of the encoder and the second-to-last layer of the decoder. We then train these new layers. This process is repeated until all the layers in the encoder and decoder have been trained. Finally, we fine-tune the entire network by training with weight updates enabled for all layers.

The layer state update, freezing, and unfreezing operations can be implemented using the following:

```Python {linenos=true}
# get references of all the layer reference of a specific class
def get_layer_refs(
    model: nn.Module,
    layer_class,
):
    # get all the fully connected layers
    # layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in model.named_modules() if isinstance(cur_module, layer_class)]
    layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in model.named_modules() if type(cur_module).__name__ == layer_class.__name__]

    return layers


# update states of dst layers from src layers
def update_corresponding_layers(
    src_layer_refs,
    dst_layer_refs,
):
    nof_src_layers = len(src_layer_refs)
    nof_dst_layers = len(dst_layer_refs)

    nof_itr_layers = min(nof_src_layers, nof_dst_layers)
    for i_layer in range(nof_itr_layers):
        cur_src_module = src_layer_refs[i_layer]["module"]
        cur_dst_module = dst_layer_refs[i_layer]["module"]
        cur_dst_module.load_state_dict(cur_src_module.state_dict())

    return nof_src_layers, nof_dst_layers


# freeze (disable grad calculation) all the layers in the input layer reference list
def freeze_layers(
    layer_refs,
):
    for cur_layer in layer_refs:
        for param in cur_layer["module"].parameters():
            param.requires_grad = False

    return layer_refs


# unfreeze (enable grad calculation) all the layers in the input layer reference list
def unfreeze_layers(
    layer_refs,    
):
    for cur_layer in layer_refs:
        for param in cur_layer["module"].parameters():
            param.requires_grad = True

    return layer_refs
```

Using these functions, we can update a deep autoencoder network from a shallower pre trained autoencoder network and manage the freezing and unfreezing of layers as follows:

```Python {linenos=true}
# get layers
src_encoder_fc_layer_refs = get_layer_refs(pretrain_encoder, torch.nn.Linear)
dst_encoder_fc_layer_refs = get_layer_refs(encoder, torch.nn.Linear)

src_decoder_fc_layer_refs = get_layer_refs(pretrain_decoder, torch.nn.Linear)
dst_decoder_fc_layer_refs = get_layer_refs(decoder, torch.nn.Linear)

src_decoder_fc_layer_refs = list(reversed(src_decoder_fc_layer_refs))
dst_decoder_fc_layer_refs = list(reversed(dst_decoder_fc_layer_refs))

## update and freeze layers
update_corresponding_layers(src_encoder_fc_layer_refs, dst_encoder_fc_layer_refs)
freeze_layers(dst_encoder_fc_layer_refs[:len(src_encoder_fc_layer_refs)])

update_corresponding_layers(src_decoder_fc_layer_refs, dst_decoder_fc_layer_refs)
freeze_layers(dst_decoder_fc_layer_refs[:len(src_decoder_fc_layer_refs)])

## unfreeze layers
unfreeze_layers(dst_encoder_fc_layer_refs)
unfreeze_layers(dst_decoder_fc_layer_refs)
```

The complete script for training the deep autoencoder can be found in the [TrainDeepSimpleFCAutoencoder notebook](https://github.com/JianZhongDev/AutoencoderPyTorch/blob/main/TrainDeepSimpleFCAutoencoder.ipynb) in my [GitHub repository](https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main).

### Tips for Autoencoder Training
- Choosing the right activation function is crucial. When using the ReLU function without careful optimization, it can lead to the 'dead ReLU' problem, causing inactive neurons in the autoencoder models.

- Avoiding a high learning rate during training, even with a scheduler (especially for autoencoders with lifetime sparsity constraints), is important. A large learning rate can cause gradient updates to overshoot in the initial epochs, potentially leading to undesired local minima during optimization.

- For training deep autoencoder networks, especially those with sparse constraints, it's beneficial to adopt a layer-by-layer iterative training approach. Training the network in stacked layers all at once can result in too few meaningful features in the latent space.

## Applications

### Compression and Dimension Reduction

The dimension reduction application of the autoencoder network is straightforward. We use the encoder network to convert high-dimensional input data into low-dimensional representations. The decoder network then reconstructs the encoded information.

After dimension reduction using the encoder, we can analyze the distribution of data in the latent space.

{{< figure src="./Images/DeeperAutoencoder_LatentSpace.png" attr="The two-dimensional codes found by a 784-128-64-32-2 fully-connected autoencoder. (image credit: Jian Zhong)" align=center target="_blank" >}}

### Denoise 

Pixel-level noise and defects cannot efficiently be represented in the much lower-dimensional latent space, so autoencoders can also be applied for noise reduction and correcting pixel defects. To train an autoencoder network for denoising, we use images with added noise as input and clean images as ground truth.

For denoising with autoencoders, we apply Gaussian noise and masking noise as data transformations in PyTorch.

The Gaussian noise transformation can be implemented as follows:

```Python {linenos=true}
# Add gaussian noise to image pixel values
class AddGaussianNoise(object):
    """
        Add gaussian noise to image pixel values
    """
    def __init__(
            self,
            mean = 0.0,
            variance = 1.0,
            generator = None,
    ):
        self.mean = mean
        self.variance = variance
        self.generator = generator # random number generator

    def __call__(self, src_image):
        src_image_shape = src_image.size()

        # generate random gaussian noise
        gauss_noise = torch.randn(
            size = src_image_shape,
            generator = self.generator,
            )
        gauss_noise = self.mean + (self.variance ** 0.5) * gauss_noise
        
        # add guassian noise to image 
        return src_image + gauss_noise

    def __repr__(self):
        return self.__class__.__name__ + f"(mean = {self.mean}, variance = {self.variance}, generator = {self.generator})"
```

Here's an example of denoising Gaussian noise using an autoencoder:

{{< figure src="./Images/GaussianNoise_50.png" attr="Gaussian denoise result of a simple fully-connected autoencoder (encoder: 784-64, decoder 64-784). (image credit: Jian Zhong)" align=center target="_blank" >}}

Masking noise involves randomly setting a fraction of pixels in the input image to zero.

```Python {linenos=true}
# Rondomly choose pixels and set them to a constant value
class RandomSetConstPxls(object):
    """
    Rondomly choose pixels and set them to a constant value
    """
    def __init__(
            self, 
            rand_rate = 0.5,
            const_val = 0,
            ):
        self.rand_rate = rand_rate
        self.const_val = const_val

    def __call__(self, src_image):
        src_image_size = src_image.size()
        tot_nof_pxls = src_image.nelement()

        # calculate number of randomly choosed pixel 
        nof_mod_pxls = tot_nof_pxls * self.rand_rate
        nof_mod_pxls = int(nof_mod_pxls)

        # generate mask for chosen pixels
        mod_pxl_mask = torch.full((tot_nof_pxls,), False)
        mod_pxl_mask[:nof_mod_pxls] = True
        mod_pxl_mask = mod_pxl_mask[torch.randperm(tot_nof_pxls)]

        # clone image and set the chosen pixels to corresponding contant value
        dst_image = src_image.clone()
        dst_image = dst_image.view(-1)
        dst_image[mod_pxl_mask] = self.const_val
        dst_image = dst_image.view(src_image_size)

        return dst_image
    
    def __repr__(self):
        return self.__class__.__name__ + f"(rand_rate = {self.rand_rate}, const_val = {self.const_val})"
```

Here's an example of using a simple fully-connected autoencoder to denoise masked noise:

{{< figure src="./Images/MaskNoise_50.png" attr="Mask denoise result of a simple fully-connected autoencoder (encoder: 784-64, decoder 64-784). (image credit: Jian Zhong)" align=center target="_blank" >}}

Refer to the [TrainSimpleDenoiseFCAutoencoder Jupyter notebook](https://github.com/JianZhongDev/AutoencoderPyTorch/blob/main/TrainSimpleDenoiseFCAutoencoder.ipynb) in my [GitHub repository](https://github.com/JianZhongDev/AutoencoderPyTorch/tree/main) for more details.

### Feature extraction and semi-supervised learning

When training an autoencoder to transform input data into a low-dimensional space, the encoder and decoder learn to map input data to a latent space and reconstruct it back. The encoder and decoder inherently capture essential features from the data through these transformations.

This feature extraction capability of autoencoders makes them highly effective for semi-supervised learning scenarios. In semi-supervised learning for classification networks, for instance, we can first train an autoencoder using the abundant unlabeled data. Subsequently, we connect a shallow fully-connected network after the encoder of the autoencoder. We then use the limited labeled data to fine-tune this shallow network.

## Reference

[1] Hinton, G. E. & Salakhutdinov, R. R. Reducing the Dimensionality of Data with Neural Networks. Science 313, 504–507 (2006).

[2] Kramer, M. A. Nonlinear principal component analysis using autoassociative neural networks. AIChE Journal 37, 233–243 (1991).

[3] Masci, J., Meier, U., Cireşan, D. & Schmidhuber, J. Stacked Convolutional Auto-Encoders for hierarchical feature extraction. in Lecture notes in computer science 52–59 (2011). doi:10.1007/978-3-642-21735-7_7.

[4] Makhzani, A. & Frey, B. J. A Winner-Take-All method for training sparse convolutional autoencoders. arXiv (Cornell University) (2014).

[5] A. Ng, “Sparse autoencoder,” CS294A Lecture notes, vol. 72, 2011.

## Citation

If you found this article helpful, please cite it as:
> Zhong, Jian (June 2024). Autoencoders with PyTorch: Full Code Guide. Vision Tech Insights. https://jianzhongdev.github.io/VisionTechInsights/posts/autoencoders_with_pytorch_full_code_guide/.

Or

```html
@article{zhong2024buildtrainAutoencoderPyTorch,
  title   = "Autoencoders with PyTorch: Full Code Guide",
  author  = "Zhong, Jian",
  journal = "jianzhongdev.github.io",
  year    = "2024",
  month   = "June",
  url     = "https://jianzhongdev.github.io/VisionTechInsights/posts/autoencoders_with_pytorch_full_code_guide/"
}
```



