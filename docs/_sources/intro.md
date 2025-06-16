# Brainhack School 2025 Project

## The Many Faces of Fear: Univariate, Predictive and Representational Perspectives on Fearful Neuroimaging

### Introduction

My name is [Shawn](https://shwnmnl.github.io) and I'm a PhD student working on bridging the gap between quantitative and qualitative approaches in cognitive neuroscience and computational psychiatry. For [Brainhack School 2025](https://school-brainhack.github.io/), my objective is to gain hands-on experience with the most popular analytic approaches to fMRI data analysis on a single subject. 

I also <s>want</s>wanted to make cool interactive brain visualizations like this one:
<div style="position: relative; width: 100%; padding-top: 56.25%;">
  <iframe src="_static/fearmap_view.html"
          style="position: absolute; top: 0; left: 0; width: 100%; height: 75%; border: none;">
  </iframe>
</div>

_Pretty cool, right?_

### Background

Discovering *"the neural signature of X"* is the bread and butter of neuroimaging research. There are myriad analytic approaches neuroscientists may deploy to this end, each with their own strengths and limitations, but none that can, on its own, deliver a singular, definitive neural signature of anything.

This limitation is both methodological and epistemological. Different methods afford different perspectives (Davis et al., 2014). Moreover, each method enables distinct types of inference about the neural underpinnings of a given construct (Popov et al., 2018).

### Objectives

Rather than seeking the neural signature of — in my case — fear, I want to compare its multiple neural signature*S*, as revealed by three complementary analytic frameworks. The goal is to compare how fear is characterized neurally depending on the method used, and to extract a high-level, integrative interpretation that acknowledges the respective strengths and limits of each approach.

Concretely, I will apply the following popular fMRI analysis frameworks:

1. Mass univariate analysis (general linear model; GLM)
2. Predictive modeling (machine learning-based decoding)
3. Representational similarity analysis (RSA)

### Data

The dataset was contributed by my colleague and labmate [Darius Valevicius](https://dariusliutas.com/), and includes fMRI recordings from participants who viewed short video clips of various animals and rated, in real time, the extent to which each clip elicited fear. The dataset is already preprocessed, including motion correction and spatial normalization. All analyses will be run on a single participant’s data as a case study.

## References
1. Davis, T., LaRocque, K. F., Mumford, J. A., Norman, K. A., Wagner, A. D., & Poldrack, R. A. (2014). *What do differences between multi-voxel and univariate analysis mean? How subject-, voxel-, and trial-level variance impact fMRI analysis.* **Neuroimage, 97, 271-283.**

2. Popov, V., Ostarek, M., & Tenison, C. (2018). *Practices and pitfalls in inferring neural representations.* **NeuroImage, 174, 340-351.**