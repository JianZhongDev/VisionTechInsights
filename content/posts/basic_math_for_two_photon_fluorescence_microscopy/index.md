---
author: "Jian Zhong"
title: "Basic Math for Two Photon Fluorescence Microscopy"
date: "2024-04-16"
description: "Basic mathematical modeling method for imaging systems, using two photon fluorescence microscopy as an example."
tags: ["modeling", "imaging system", "optics"]
categories: ["optics", "modeling"]
series: ["point scan imaging"]
aliases: ["basic-math-2PFM"]
cover:
   image: images/basic_math_for_two_photon_fluorescence_microscopy/2PFM_Principles.png
   caption: "principles of two photon fluorescence microscopy (image credit: Jian Zhong)"
ShowToc: true
TocOpen: false
math: true
ShowBreadCrumbs: true
---


It's really helpful to do some basic math to understand how well an imaging system works when you're designing or improving it. Most of the time, a very basic model of the system can provide a ton of useful information. Therefore, I would like to share an example of how we can crunch some numbers to understand the signal and noise in a two-photon fluorescence microscope (2PFM). In the end, I will also provide a practical demo of how this math can help us make the system work better.

## Introduction to Two-Photon Microscopy

<!-- ![2PFM Principle](./2PFM_Principles.png)
*Light path and principles of two photon fluorescence microscopy (image credit: Jian Zhong)* -->

Two-photon fluorescence microscopes work by scanning laser beams across a sample and capturing the fluorescent light emitted at each spot. Typically, they use scanning mirrors like galvo mirrors to move the laser beams around. At each scan location, laser beam reflected by the scanning mirrors is relayed to the objective lens using a tube lens and a scan lens. The objective lens then focuses the laser beam into the sample. Afterward, the fluorescence emitted from the sample is gathered by the same objective lens, redirected by a dichroic mirror (which transmits the excitation light while reflects the fluorescence), and finally gathered by some collection optics before being detected by a photodetector (usually a photomultiplier tube (PMT)).

Two photon fluorescence microscopes utilize two-photon excitation phenomenon to make fluorophores in samples light up. This happens when the molecules absorb two photons of light at the same time. Because of its low possibility, two photon excitation only occurs in the focal point inside the sample, where the laser beam is most intense (the density of the excitation photons reaches a maximum). To increase the efficiency of two photon excitation and avoid damaging the sample, two photon microscopes send laser pulses with very short pulse width (~femtosecond) into the sample, instead of using a high power continuous-wave laser . 

## Modeling of the Imaging System

### Excitation Process

### Two Photon Absorption and Fluorescence Generation

### Fluorescence Signal Detection  

### Integration of Multiple Pulses and Signal-to-Noise Ratio

### Side Notes

## Example Application

## Conclusion

## Citation

If you found this article helpful, please cite it as:
> Zhong, Jian (Apr 2024). Basic Math for Two Photon Fluorescence Microscopy. Vision Tech Insights. https://jianzhongdev.github.io/VisionTechInsights/posts/basic_math_for_two_photon_fluorescence_microscopy/.

Or

```html
@article{zhong2024basicmath2PFM,
  title   = "Basic Math for Two Photon Fluorescence Microscopy",
  author  = "Zhong, Jian",
  journal = "jianzhongdev.github.io",
  year    = "2024",
  month   = "Apr",
  url     = "https://jianzhongdev.github.io/VisionTechInsights/posts/basic_math_for_two_photon_fluorescence_microscopy/"
}
```





Mathematical notation in a Hugo project can be enabled by using third party JavaScript libraries.

<!--more-->

In this example we will be using [KaTeX](https://katex.org/)

-   Create a partial under `/layouts/partials/math.html`
-   Within this partial reference the [Auto-render Extension](https://katex.org/docs/autorender.html) or host these scripts locally.
-   Include the partial in your templates ([`extend_head.html`](../papermod/papermod-faq/#custom-head--footer)) like so:
-   refer [ISSUE #236](https://github.com/adityatelange/hugo-PaperMod/issues/236)

```bash
{{ if or .Params.math .Site.Params.math }}
{{ partial "math.html" . }}
{{ end }}
```

-   To enable KaTex globally set the parameter `math` to `true` in a project's configuration
-   To enable KaTex on a per page basis include the parameter `math: true` in content files

**Note:** Use the online reference of [Supported TeX Functions](https://katex.org/docs/supported.html)

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}

### Examples

{{< math.inline >}}

<p>
Inline math: \(\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887…\)
</p>
{{</ math.inline >}}

Block math:

$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } }
$$
