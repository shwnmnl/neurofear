<p align="center">
    <img src="fearfaces.png" alt="drawing" width="500"/>
</p>

# The Many Faces of Fear: Univariate, Predictive and Representational Perspectives on Fearful Neuroimaging

I am a PhD student working on bridging the gap between quantitative and qualitative approaches in cognitive neuroscience and computational psychiatry. During my master's, I worked on mapping out subjective experience using a combination of psychometrics, NLP and generative AI, with a focus on accelerating mental health care research. For Brainhack School 2025, I want to gain hands-on experience with the most popular analytic approaches to neuroimaging on a single subject. 

## Background
Discovering *"the neural signature of X"* is the bread and butter of neuroimaging research. There are myriad analytic approaches neuroscientists may deploy to this end, each with their own strengths and limitations, but none that can, on its own, deliver a singular, definitive neural signature of ***anything***. 

This limitation is both methodological and epistemological. Different methods afford different perspectives ([Davis et al., 2014](#1-davis-t-larocque-k-f-mumford-j-a-norman-k-a-wagner-a-d--poldrack-r-a-2014-what-do-differences-between-multi-voxel-and-univariate-analysis-mean-how-subject--voxel--and-trial-level-variance-impact-fmri-analysis-neuroimage-97-271-283)). Moreover, each method enables distinct types of inference about the neural underpinnings of a given construct ([Popov et al., 2018](#2-popov-v-ostarek-m--tenison-c-2018-practices-and-pitfalls-in-inferring-neural-representations-neuroimage-174-340-351)). 

## Objectives
Rather than seeking ***the*** neural signature of—-in my case—-fear, I want to compare its multiple neural signature***S***, as revealed by three complementary analytic frameworks. The goal is to compare how fear is characterized neurally depending on the method used, and to extract a high-level, integrative interpretation that acknowledges the respective strengths and limits of each approach.

Concretely, I will apply the following popular fMRI analysis frameworks:
- Mass univariate analysis (general linear model; GLM)
- Predictive modeling (machine learning-based decoding)
- Representational similarity analysis (RSA)

All analyses will be run on a single participant’s data as a case study, with the possibility of limited comparisons to other participants if useful.

## Data
The dataset was contributed by my colleague and labmate [Darius Valevicius](https://dariusliutas.com/), and includes fMRI recordings from participants who viewed short video clips of various animals and rated, in real time, the extent to which each clip elicited fear. The dataset is already preprocessed, including motion correction and spatial normalization. 

## Tools

<u>Planned tools and packages include:</u>
- git and github for version control and documentation
- nibabel/nilearn for NIfTI image loading and saving, GLM fitting, decoding, and visualization
- scipy for statistical functions, distance metrics (for RSA)
- scikit-learn for predictive modeling, cross-validation, hyperparameter tuning
- scipy and pandas for data manipulation and RSA
- matplotlib/plotly for plotting
- Jupyter Book for interactive documentation and results dissemination


## Deliverables

- Github repo
- Jupyter notebooks
- Jupyter Book site
- [Slide deck](https://www.canva.com/design/DAGoggkx5Qc/0rdEuKNYpgVpxGgaHn7XuA/edit?) and report presenting side-by-side outputs and interpretations

## References
#### 1. Davis, T., LaRocque, K. F., Mumford, J. A., Norman, K. A., Wagner, A. D., & Poldrack, R. A. (2014). *What do differences between multi-voxel and univariate analysis mean? How subject-, voxel-, and trial-level variance impact fMRI analysis.* **Neuroimage, 97, 271-283.**

#### 2. Popov, V., Ostarek, M., & Tenison, C. (2018). *Practices and pitfalls in inferring neural representations.* **NeuroImage, 174, 340-351.**
