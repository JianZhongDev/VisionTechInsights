---
author: "Jian Zhong"
title: "A Gentle Introduction to Variational Autoencoders: Concept and PyTorch Implementation Guide"
date: "2024-07-08"
description: "A comprehensive guide on the concepts and PyTorch implementation of variational autoencoder."
tags: ["computer vision", "machine learning"]
categories: ["computer vision", "modeling"]
series: ["computer vision"]
aliases: ["gentle_introduction_to_variational_autoencoders"]
cover:
   image: images/gentle_introduction_to_variational_autoencoders/VariationalAutoEncoderStructure.png
   caption: "[cover image] Architecture of Variational Autoencoder (image credit: Jian Zhong)"
ShowToc: true
TocOpen: false
math: true
ShowBreadCrumbs: true
---

The variational autoencoder (VAE) is a type of generative model that combines principles from neural networks and probabilistic models to learn the underlying probabilistic distribution of a dataset and generate new data samples similar to the given dataset. 

Due to its ability to combine probabilistic modeling and learn complex data distributions, VAEs have become a fundamental tool and have had a profound impact on the fields of machine learning and deep learning. 

In this blog post, we will start with a quick introduction to the architecture of variational autoencoders and a comparison between variational autoencoders and conventional autoencoders. Next, we will use mathematical expressions and graphics to explain the concepts behind the variational autoencoder network design. Lastly, we will provide a step-by-step tutorial on how to build and train a variational autoencoder network using PyTorch. All the code and demonstration scripts can be found in [my VAE GitHub repository](https://github.com/JianZhongDev/VariationalAutoencoderPytorch): (URL https://github.com/JianZhongDev/VariationalAutoencoderPytorch )


## Variational Autoencoder Structures

A variational autoencoder (VAE) model usually includes an encoder network, a distribution model, and a decoder network.

The encoder network learns how to transform each data point in the dataset into the parameters of a probabilistic distribution in the latent space.

The distribution model uses these parameters to create the probabilistic distribution and draws random variables from this distribution in the latent space.

The decoder network then learns how to transform these latent space random variables back into their corresponding data points in the original dataset space.

A forward pass through the entire VAE works like this: the encoder processes the input data and generates the parameters for the latent space distribution. The distribution model then creates the latent space probabilistic distribution based on these parameters and draws a random variable from it. Finally, the decoder reconstructs the original data from this latent space random variable.

{{< figure src="./Images/VariationalAutoEncoderStructure.png" attr="Structure of a variational autoencoder. (image credit: Jian Zhong)" align=center target="_blank" >}}

## Variational Autoencoder and Autoencoder

The variational autoencoder (VAE) has a similar encoder-decoder structure to conventional autoencoders, but the functions of the encoders and decoders are very different.

In conventional autoencoders, the encoder and decoder networks learn direct transformations between the data space and the latent space. The encoder maps data points to a specific point in the latent space, and the decoder maps these points back to the original data space.

{{< figure src="./Images/ConventionalAutoEncoder.png" attr="Conventional autoencoder. (image credit: Jian Zhong)" align=center target="_blank" >}}

In contrast, a VAE learns to transform data into a probabilistic distribution in the latent space, rather than specific points. The encoder maps data points to the parameters of a probabilistic distribution. The distribution model then draws random variables from this distribution. The decoder transforms these random variables back into the data space.

{{< figure src="./Images/VariationalAutoEncoder.png" attr="Variational autoencoder. (image credit: Jian Zhong)" align=center target="_blank" >}}


## Math of Variational Autoencoder

This section explains the concepts behind variational autoencoders and walks through their mathematical derivation. There will be a lot of math here. If you prefer a quick understanding of how variational autoencoders work and find this section too math-heavy, feel free to skip ahead to the graphical explanation.
Please note that the mathematical expressions in this section are meant to convey the basic ideas behind variational autoencoders, so they might not be very rigorous. For a more detailed and rigorous mathematical explanation, please refer to the [variational autoencoder paper](https://arxiv.org/abs/1312.6114).

### Problem Description


Let's consider a dataset   {{< math.inline >}} \( X = \{x^{(i)}\}_{i=1}^{N} \) {{</ math.inline >}} ​ consisting of {{< math.inline >}} \( N \) {{</ math.inline >}} independent and identically distributed (i.i.d.) samples. The probabilistic distribution of these samples is described as  {{< math.inline >}} \( p_{\theta}(x) \) {{</ math.inline >}}, where  {{< math.inline >}} \( \theta \) {{</ math.inline >}} are the parameters defining this distribution. The {{< math.inline >}} \( \theta \) {{</ math.inline >}} here is a general description of the potential parameters for the distribution and may or may not include our model's parameters.

From a probabilistic perspective, learning is about finding the optimal parameters {{< math.inline >}} \( \theta^{*} \) {{</ math.inline >}} for the distribution {{< math.inline >}} \( p_{\theta}(x) \) {{</ math.inline >}} so that the probability of each data point xxx in the learning dataset is maximized.

The variational autoencoder addresses a specific part of this learning problem, where the process of generating data xxx depends on an unobserved random variable {{< math.inline >}} \( z \) {{</ math.inline >}}. To generate the random variable {{< math.inline >}} \( x \) {{</ math.inline >}}, a random variable {{< math.inline >}} \( z \) {{</ math.inline >}} is first generated from a prior distribution  {{< math.inline >}} \( p_{\theta}(z) \) {{</ math.inline >}}, and then {{< math.inline >}} \( x \) {{</ math.inline >}} is generated from a conditional distribution {{< math.inline >}} \( p_{\theta}(x|z) \) {{</ math.inline >}}. Using Bayes's theorem, the probabilistic distribution {{< math.inline >}} \( p_{\theta}(x) \) {{</ math.inline >}} can be expressed as follows:

$$
p_{\theta}(x) = \frac{p_{\theta}(x|z)}{p_{\theta}(z|x)} p_{\theta}(z)
$$

The variational autoencoder aims to learn and model {{< math.inline >}} \( p_{\theta}(z) \) {{</ math.inline >}}, {{< math.inline >}} \( p_{\theta}(z|x) \) {{</ math.inline >}}, and {{< math.inline >}} \( p_{\theta}(x|z) \) {{</ math.inline >}} to maximize {{< math.inline >}} \( p_{\theta}(x) \) {{</ math.inline >}} after the learning process.

### Maximizing Probability and Lower Bound

Usually, modeling  {{< math.inline >}} \( p_{\theta}(z|x) \) {{</ math.inline >}} exactly is not feasible. So, we use an approximate model  {{< math.inline >}} \( q_{\phi}(z|x) \) {{</ math.inline >}}, where {{< math.inline >}} \( \phi \) {{</ math.inline >}} represents the model parameters.

Given a dataset {{< math.inline >}} \( X = \{x^{(i)}\}_{i=1}^{N} \) {{</ math.inline >}} of the random variable {{< math.inline >}} \( x \) {{</ math.inline >}}, maximizing {{< math.inline >}} \( p_{\theta}(x) \) {{</ math.inline >}} means maximizing the likelihood of the data points, {{< math.inline >}} \( p_{\theta}(x^{(1)}, ..., x^{(N)} ) \) {{</ math.inline >}}. This is the same as maximizing  {{< math.inline >}} \( log(p_{\theta}(x^{(1)}, ..., x^{(N)} )) \) {{</ math.inline >}}. Since all data points are i.i.d., we have:

$$
log(p_{\theta}(x^{(1)}, ..., x^{(N)} )) = log(\prod_{i=1}^{N} x^{(i)}) = \sum_{i=1}^{N} log(x^{(i)})
$$

Using the KL divergence between {{< math.inline >}} \( q_{\phi}(z|x^{(i)}) \) {{</ math.inline >}} and  {{< math.inline >}} \( p_{\theta}(z|x^{(i)}) \) {{</ math.inline >}}, we can rewrite {{< math.inline >}} \( log(p_{\theta}(x^{(i)}) ) \) {{</ math.inline >}}:

$$
D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})) = \sum_{z}( q_{\phi}(z|x^{(i)}) \cdot log(\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z|x^{(i)})}) ) 
$$

Expanding and rearranging, we get:

$$
D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)}))  = \sum_{z}( q_{\phi}(z|x^{(i)}) \cdot log(\frac{q_{\phi}(z|x^{(i)}) p_{\theta}(x^{(i)})}{p_{\theta}(x^{(i)}|z) p_{\theta}(z)}) )
$$
$$
 = \sum_{z}( q_{\phi}(z|x^{(i)}) \cdot log(\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)})) +  \sum_{z}( q_{\phi}(z|x^{(i)}) \cdot log(p_{\theta}(x^{(i)}))) - \sum_{z}( q_{\phi}(z|x^{(i)}) \cdot  p_{\theta}(x^{(i)}|z))
$$
$$
= \sum_{z}( q_{\phi}(z|x^{(i)}) \cdot log(\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)})) + log(p_{\theta}(x^{(i)})) - \mathbb{E_{z}}(  log(p_{\theta}(x^{(i)}|z)) )
$$

So,

$$
log(p_{\theta}(x^{(i)})) = D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)}))  + \mathcal{L}(\theta, \phi, x^{(i)})
$$

where,

$$
\mathcal{L}(\theta, \phi, x^{(i)}) = - \sum_{z}( q_{\phi}(z|x^{(i)}) \cdot log(\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)})) + \mathbb{E_{z}}( log(p_{\theta}(x^{(i)}|z)) )
$$
$$
= - D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z)) + \mathbb{E_{z}}( log(p_{\theta}(x^{(i)}|z)) )
$$

According to the definition of KL divergence, we know 

$$
D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})) \geq 0
$$

Therefore,

$$
log(p_{\theta}(x^{(i)})) \geq \mathcal{L}(\theta, \phi, x^{(i)})
$$

In other words,  {{< math.inline >}} \( \mathcal{L}(\theta, \phi, x^{(i)}) \) {{</ math.inline >}} is the lower bound of  {{< math.inline >}} \( log(p_{\theta}(x^{(i)})) \) {{</ math.inline >}}. If we maximize {{< math.inline >}} \( \mathcal{L}(\theta, \phi, x^{(i)}) \) {{</ math.inline >}}  for each {{< math.inline >}} \(  x^{(i)} \) {{</ math.inline >}}, we effectively maximize {{< math.inline >}} \(  log(p_{\theta}(x^{(1)}, ..., x^{(N)} )) \) {{</ math.inline >}}, which is our learning objective.

During training, we typically define a loss function to minimize. Based on the above discussion, the loss function for the variational autoencoder can be set up as:

$$
Loss = \sum_{i=1}^{N}(-\mathcal{L}(\theta, \phi, x^{(i)}))
$$

Now, let's take a closer look at the terms in  {{< math.inline >}} \( \mathcal{L}(\theta, \phi, x^{(i)}) \) {{</ math.inline >}}:

$$
\mathcal{L}(\theta, \phi, x^{(i)}) = - D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z)) + \mathbb{E_{z}}( log(p_{\theta}(x^{(i)}|z)) )
$$

Within this expression, the probabilistic distribution {{< math.inline >}} \( q_{\phi}(z|x^{(i)}) \) {{</ math.inline >}}, {{< math.inline >}} \(p_{\theta}(z) \) {{</ math.inline >}}, {{< math.inline >}} \( p_{\theta}(x^{(i)}|z) \) {{</ math.inline >}} need to be modeled and learned. In a variational autoencoder, {{< math.inline >}} \( q_{\phi}(z|x^{(i)}) \) {{</ math.inline >}} is typically learned using the encoder network with a predefined probabilistic model. {{< math.inline >}} \(p_{\theta}(z) \) {{</ math.inline >}} is usually given by prior knowledge or assumptions. {{< math.inline >}} \( p_{\theta}(x^{(i)}|z) \) {{</ math.inline >}}  is learned by the decoder network.

The term {{< math.inline >}} \( \mathbb{E_{z}}( log(p_{\theta}(x^{(i)}|z)) ) \) {{</ math.inline >}} measures the probability of sample {{< math.inline >}} \( x^{(i)} \) {{</ math.inline >}} across all possible {{< math.inline >}} \( z \) {{</ math.inline >}} values, which also indicates how similar the encoder's reconstruction is to the original data. This term corresponds to the reconstruction error term in the autoencoder's loss function. The term {{< math.inline >}} \( D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z)) \) {{</ math.inline >}} measures the similarity between {{< math.inline >}} \( q_{\phi}(z|x^{(i)}) \) {{</ math.inline >}} and {{< math.inline >}} \(p_{\theta}(z) \) {{</ math.inline >}}, and acts as a regularization term in the total loss function.

### Adding Prior Knowledge or Assumptions About the Distributions

Now that we've established the relationships within the loss function and the components of variational autoencoders, we're ready to incorporate prior knowledge or assumptions into the model and proceed with the learning process.

For instance, we can introduce the following Gaussian-based assumptions to the variational autoencoder:

- For {{< math.inline >}} \( q_{\phi}(z|x^{(i)}) \) {{</ math.inline >}}, we model it as a diagonal multivariate Gaussian distribution {{< math.inline >}} \(\mathcal{N}(\mu^{(i)}, \sigma^{(i)}) \) {{</ math.inline >}} for each input data point {{< math.inline >}} \( x^{(i)} \) {{</ math.inline >}}. The encoder learns to map each sample from the dataset space to the parameters (i.e., {{< math.inline >}} \( \mu^{(i)} \) {{</ math.inline >}} and {{< math.inline >}} \( \sigma^{(i)} \) {{</ math.inline >}} ) of its corresponding Gaussian distribution in the latent space.

- {{< math.inline >}} \(p_{\theta}(z) \) {{</ math.inline >}} is simply modeled as a unit normal distribution {{< math.inline >}} \(\mathcal{N}(0, 1) \) {{</ math.inline >}}

- {{< math.inline >}} \( p_{\theta}(x^{(i)}|z) \) {{</ math.inline >}} is modeled such that the decoder learns to map {{< math.inline >}} \( z \) {{</ math.inline >}} to {{< math.inline >}} \( \hat{x} \) {{</ math.inline >}}, where {{< math.inline >}} \( \hat{x} = D_{\theta}(z) \) {{</ math.inline >}} represents the reconstructed data in the dataset space. Here, {{< math.inline >}} \( D_{\theta}(z) \) {{</ math.inline >}} denotes the output of the decoder, ensuring {{< math.inline >}} \( p_{\theta}(x^{(i)}|z) \) {{</ math.inline >}}  follows a Gaussian distribution {{< math.inline >}} \(\mathcal{N}(D_{\theta}(z), \sigma) \) {{</ math.inline >}}

With this prior knowledge, the KL divergence term using diagonal Gaussian and unit Gaussian can be expressed as:

$$
 D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})) = \frac{1}{2} \sum_{j=1}^{J}(-1 - log( (\sigma_{j}^{(i)})^{2}) + (\mu_{j}^{(i)})^{2} + (\sigma_{j}^{(i)})^{2})
$$

(NOTE: refer *examples* section of the [LK Divergence Wikipedia page](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) for the expressoin above)

Here,  {{< math.inline >}} \( j \) {{</ math.inline >}} indexes the dimensions of the latent space Gaussian distribution, and  {{< math.inline >}} \( J \) {{</ math.inline >}} specifies the dimensionality, which is set when building the variational autoencoder model.

The term  {{< math.inline >}} \( \mathbb{E_{z}}( log(p_{\theta}(x^{(i)}|z)) ) \) {{</ math.inline >}} can be written as:

$$
\mathbb{E_{z}}( log(p_{\theta}(x^{(i)}|z)) ) = \mathbb{E_{z}}(log( A \cdot exp( - \frac{(x - D_{\theta}(z))^{2}}{2 \sigma^{2}}  ) )) 
$$
$$
= log(A) -\frac{1}{2 \sigma^{2}} \mathbb{E_{z}}((x - D_{\theta}(z))^{2})
$$
$$
= log(A) -\frac{1}{2 \sigma^{2}} \sum_{i=1}^{N}((x^{(i)} - \hat{x}^{(i)})^{2})
$$

Here, {{< math.inline >}} \( A \) {{</ math.inline >}} is the normalization constant for the proposed Gaussian distribution, which is independent of model parameters and can be disregarded during optimization. {{< math.inline >}} \( \sigma \) {{</ math.inline >}} specified value when constructing the variational autoencoder model, adjusting how distinct each data reconstruction should be and balancing the weights of the reconstruction error loss term and the normal distribution regularization term.

Furthermore, the lower bound {{< math.inline >}} \( \mathcal{L}(\theta, \phi, x^{(i)}) \) {{</ math.inline >}} can be approximated as:

$$
\mathcal{L}(\theta, \phi, x^{(i)}) \simeq \frac{1}{2} \sum_{j=1}^{J}(1 + log( (\sigma_{j}^{(i)})^{2}) - (\mu_{j}^{(i)})^{2} - (\sigma_{j}^{(i)})^{2}) -\frac{1}{2 \sigma^{2}} \sum_{i=1}^{N}((x^{(i)} - \hat{x}^{(i)})^{2})
$$

Finally, the loss function can be defined as:

$$
Loss = \sum_{i=1}^{N}(-\mathcal{L}(\theta, \phi, x^{(i)}))
$$
$$
= \sum_{i=1}^{N}(\frac{1}{2 \sigma^{2}} \sum_{i=1}^{N}((x^{(i)} - \hat{x}^{(i)})^{2})) + \sum_{i=1}^{N}(\frac{1}{2} \sum_{j=1}^{J}(-1 - log( (\sigma_{j}^{(i)})^{2}) + (\mu_{j}^{(i)})^{2} + (\sigma_{j}^{(i)})^{2}) )
$$

These assumptions help solidify each component of the variational autoencoder. The encoder is a neural network that takes data as input and outputs parameters {{< math.inline >}} \( \mu^{(i)} \) {{</ math.inline >}} and {{< math.inline >}} \( \sigma^{(i)} \) {{</ math.inline >}} of a diagonal distribution in the latent space. In the latent space, a diagonal multivariate Gaussian distribution is created based on the encoder's parameters, and a random variable sample {{< math.inline >}} \( z^{(i)} \) {{</ math.inline >}} is sampled from this distribution. The decoder, another neural network, takes the latent space variable as input and produces a reconstruction of the data in the dataset space.


### Graphical Explanation of Variational Autoencoder Learning Process

A variational autoencoder comprises an encoder, a distribution model, and a decoder. In processing each sample from the dataset during a forward pass, the encoder first transforms the data into a corresponding probability distribution in the latent space. Then, a random variable is drawn from this distribution in the latent space. Finally, the decoder uses this random variable to reconstruct the distribution in the dataset space.

During training, each dataset sample serves as input to the variational autoencoder and as the target for comparing the difference between the encoder's output and the original input data.

Through this process, the encoder adjusts by pushing the predicted distributions of latent space apart for samples with distinct features and pulling them closer together for samples with similar features. Consequently, random variables drawn from input samples with similar features tend to be close to each other in the latent space, while those from dissimilar samples are farther apart.

Moreover, as the learning progresses, latent space variables from similar input features move closer together, leading to similar reconstructed representations in the data space. Conversely, variables from different input features move farther apart, resulting in distinct representations in the data space.

{{< figure src="./Images/VariationalAutoEncoderLearning.png" attr="Learn process of variational autoencoder. (image credit: Jian Zhong)" align=center target="_blank" >}}

## Building a Variational Autoencoder with PyTorch

Starting from this point onward, we will use the variational autoencoder with the Gaussian modeling prior knowledge we discussed earlier to demonstrate how to build and train a variational autoencoder using PyTorch.

Please refer to the [TrainSimpleGaussFCVAE](https://github.com/JianZhongDev/VariationalAutoencoderPytorch/blob/main/TrainSimpleGaussFCVAE.ipynb) notebook in [my GitHub repository](https://github.com/JianZhongDev/VariationalAutoencoderPytorch/tree/main) for the complete training notebook.

As mentioned earlier, here is how you can define the encoder, distribution model, and decoder:


```Python {linenos=true}
import torch
from torch import nn

from .Layers import (DebugLayers, StackedLayers)


# Encoder learns mappling from input data x to latent z gaussian distribution parameters (i.e. mu and sigma)
class SimpleGaussVAEFCEncoder(nn.Module):
    def __init__(
        self,
        prev_layer_descriptors = [],
        gaussparam_layer_descriptors = [],
    ):
        assert isinstance(prev_layer_descriptors, list)
        assert isinstance(gaussparam_layer_descriptors, list)
        super().__init__()

        self.prev_network = StackedLayers.VGGStackedLinear(prev_layer_descriptors)

        in_features_key = r"in_features"
        if in_features_key not in gaussparam_layer_descriptors[0]:
            last_fc_idx = len(self.prev_network.network) - 1
            while(last_fc_idx > 0):
                if(isinstance(self.prev_network.network[last_fc_idx], nn.Linear)):
                    break
                last_fc_idx -= 1
            gaussparam_layer_descriptors[0][in_features_key] = self.prev_network.network[last_fc_idx].weight.size(0)

        self.gauss_mu_network = StackedLayers.VGGStackedLinear(gaussparam_layer_descriptors)
        self.gauss_logsigma_network = StackedLayers.VGGStackedLinear(gaussparam_layer_descriptors)

    def forward(self, x):
        prev_y = self.prev_network(x)
        gauss_mu = self.gauss_mu_network(prev_y)
        gauss_sigma = torch.exp(self.gauss_logsigma_network(prev_y))
        return (gauss_mu, gauss_sigma)


# generate diagonal gaussian random variables
class DiagGaussSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_normal = torch.distributions.normal.Normal(0, 1)

    def forward(self, gauss_paras):
        mu, sigma = gauss_paras
        normal_samples = self.unit_normal.sample(sigma.size()).to(sigma.device)
        z = mu + sigma * normal_samples
        return z

# decoder learns mapping from latent variable z to reconstructed data x_hat 
class SimpleGaussVAEFCDecoder(nn.Module):
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


## Code to create models
latent_dim = 2

## create encoder model
encoder_prev_layer_descriptors = [
    {"nof_layers": 1, "in_features": nof_features, "out_features": 400, "activation": torch.nn.LeakyReLU},
]
encoder_gaussparam_layer_descriptors = [
    {"nof_layers": 1, "out_features": latent_dim, "activation": None},
]

encoder = SimpleGaussVAEFCEncoder(
    prev_layer_descriptors = encoder_prev_layer_descriptors,
    gaussparam_layer_descriptors = encoder_gaussparam_layer_descriptors,
)

## create latent space distribution generation model
distrib_sample = DiagGaussSample()


## create decoder model
decoder_layer_descriptors = [
    {"nof_layers": 1, "in_features": latent_dim, "out_features": 400, "activation": torch.nn.LeakyReLU},
    {"nof_layers": 1, "out_features": nof_features, "activation": torch.nn.Sigmoid},
]

decoder = SimpleGaussVAEFCDecoder(
    layer_descriptors = decoder_layer_descriptors
)
```

The VGGStackedLinear module creates several fully-connected networks based on the input layer descriptors. For a detailed explanation, please refer to my blog post on [building and training VGG network with PyTorch](../implement_train_VGG_PyTorch/index.md).

And here's how you can implement a forward pass of the autoencoder:

```Python {linenos=true}
features, targets = next(iter(dataloader))
distrib_params = encoder(features)
codes = distrib_sample(distrib_params)
preds = decoder(codes)
```

## Training and Evaluating a Variational Autoencoder with PyTorch

### Loss function

Based on the discussion about the loss function, we can easily implement the verification loss for the decoder reconstruction result like this:

```Python {linenos=true}
# nn.Module for similarity loss wih Gaussian assumptions
class GaussSimilarityLoss(nn.Module):
    def __init__(
            self,
            gauss_sigma = 1.0,
    ):
        assert gauss_sigma > 0
        super().__init__()
        self.register_buffer("gauss_sigma", torch.tensor(gauss_sigma))

    def forward(self, x, x_hat):
        x = torch.flatten(x, start_dim = 1, end_dim = -1)
        x_hat = torch.flatten(x_hat, start_dim = 1, end_dim = -1)
        batch_loss = 1/(2* (self.gauss_sigma**2) ) * torch.sum( (x - x_hat)**2, dim = -1)
        loss = torch.mean(batch_loss)
        return loss
```

We can also implement the Gaussian prior regularization term like this:

```Python {linenos=true}
# Kullback-Leibler divergence formula for diagnomal gauss distribution
def unitgauss_kullback_leibler_divergence(
        gauss_paras    
):
    mu, sigma = gauss_paras
    d_kl = 0.5 * torch.sum(sigma**2 + mu**2 - 1 - 2 * torch.log(sigma), dim = -1)
    return d_kl
    

# nn.Module for Kullback-Leibler divergence with unit gauss distribution
class UnitGaussKullbackLeiblerDivergenceLoss(nn.Module):
    def forward(self, gauss_params):
        d_kl = torch.mean(unitgauss_kullback_leibler_divergence(gauss_params))
        return d_kl

```

### Train and validate one epoch

The script to train and validate the auto encoder model for one epoch can be implemented as follows:

```Python {linenos=true}
# train encoder and decoder for one epoch
def train_one_epoch(
    encoder_model,
    distrib_model, 
    decoder_model,
    train_loader,
    data_loss_func,
    optimizer,
    distrib_loss_rate = 0,
    distrib_loss_func = None,
    device = None,
):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = 0

    encoder_model.train(True)
    decoder_model.train(True)
    distrib_model.train(True)

    for i_batch, data in enumerate(train_loader):        
        inputs, targets = data

        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()
        
        cur_distrib_params = encoder_model(inputs) # encode input data into latent space distribution 
        cur_codes = distrib_model(cur_distrib_params) # generate random variables in distribtion space
        cur_preds = decoder_model(cur_codes) # reconstruct input image
        
        data_loss = data_loss_func(cur_preds, targets) 

        # loss for contraints in the latent space
        distrib_loss = 0
        if distrib_loss_func:
            distrib_loss = distrib_loss_func(cur_distrib_params)
        loss = data_loss + distrib_loss_rate * distrib_loss
        
        loss.backward()
        optimizer.step()
        
        tot_loss += loss.item()
        tot_nof_batch += 1

        if i_batch % 100 == 0:
            print(f"batch {i_batch} loss: {tot_loss/tot_nof_batch: >8f}")

    avg_loss = tot_loss/tot_nof_batch

    print(f"Train: Avg loss: {avg_loss:>8f}")

    return avg_loss

# validate encoder and decoder for one epoch
def validate_one_epoch(
    encoder_model,
    distrib_model, 
    decoder_model,
    validate_loader,
    data_loss_func,
    distrib_loss_rate = 0,
    distrib_loss_func = None,
    device = None,
):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = 0

    encoder_model.eval()
    decoder_model.eval()
    distrib_model.eval()

    for i_batch, data in enumerate(validate_loader):        
        inputs, targets = data

        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)
        
        cur_distrib_params = encoder_model(inputs) # encode input data into latent space distribution 
        cur_codes = distrib_model(cur_distrib_params) # generate random variables in distribtion space
        cur_preds = decoder_model(cur_codes) # reconstruct input image
        
        data_loss = data_loss_func(cur_preds, targets) 

        # loss for contraints in the latent space
        distrib_loss = 0
        if distrib_loss_func:
            distrib_loss = distrib_loss_func(cur_distrib_params)
        loss = data_loss + distrib_loss_rate * distrib_loss
        
        tot_loss += loss.item()
        tot_nof_batch += 1

        if i_batch % 100 == 0:
            print(f"batch {i_batch} loss: {tot_loss/tot_nof_batch: >8f}")

    avg_loss = tot_loss/tot_nof_batch

    print(f"Validate: Avg loss: {avg_loss:>8f}")

    return avg_loss
```

## Learning Results of a Variational Autoencoder

Finally, we can check the results produced by the variational autoencoder. Here is the distribution of the latent space random variables drawn from the Gaussian distribution of the MNIST testing dataset:

{{< figure src="./Images/VAE_LatentSpace.png" attr="Learn latent space distribution. (image credit: Jian Zhong)" align=center target="_blank" >}}

When we compare this to the latent space distribution from a conventional autoencoder (check [my autoencoder blog](https://jianzhongdev.github.io/VisionTechInsights/posts/autoencoders_with_pytorch_full_code_guide/) post for the comparison result), we see that the variational autoencoder's latent space distribution is more Gaussian. This is expected because we included Gaussian distribution modeling as prior knowledge when building the variational autoencoder.

We can also generate a manifold of the learned decoder by adjusting the latent space variable values continuously and using the decoder to produce their reconstructions in the dataset space.

{{< figure src="./Images/VAE_Manifold_X-2_2_Y_-2_2.png" attr="learned manifold. (image credit: Jian Zhong)" align=center target="_blank" >}}

## Reference

[1] Diederik P. Kingma and Max Welling. 2013. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114 (2013).

## Citation

If you found this article helpful, please cite it as:
> Zhong, Jian (July 2024). A Gentle Introduction to Variational Autoencoders: Concept and PyTorch Implementation Guide. Vision Tech Insights. https://jianzhongdev.github.io/VisionTechInsights/posts/gentle_introduction_to_variational_autoencoders/.

Or

```html
@article{zhong2024GentleIntroVAE,
  title   = "A Gentle Introduction to Variational Autoencoders: Concept and PyTorch Implementation Guide",
  author  = "Zhong, Jian",
  journal = "jianzhongdev.github.io",
  year    = "2024",
  month   = "July",
  url     = "https://jianzhongdev.github.io/VisionTechInsights/posts/gentle_introduction_to_variational_autoencoders/"
}
```



