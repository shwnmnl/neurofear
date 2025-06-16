# Approach 3 - Representational 

Representational Similarity Analysis (RSA) assesses whether the pattern of neural responses across conditions within a given brain region mirrors the structure of behavioral or task-related variables. Here, we compute pairwise dissimilarities between trials based on regional brain activity and compare this neural representational geometry to that derived from fear ratings. In other words, we want to know whether the structure of fear ratings across trials is reflected in trial-wise activity patterns in specific brain regions.

## Imports

```
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.input_data import NiftiLabelsMasker
```

## Load atlas and define ROI masker

```
atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_img = atlas.maps
labels = atlas.labels

masker_roi = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True,
    detrend=True,
    t_r=0.867
)
```

## Get trial-wise ROI activations

```
masker_roi.fit(betas)
X_roi = masker_roi.transform(betas)  # shape: (n_trials, n_regions)
```

## Compute fear RDM

```
fear_dist = pdist(y_fear[:, None], metric='cityblock')
RDM_fear = squareform(fear_dist)
```

## Compute ROI brain RDMs and correlate with fear RDMs

```
n_regions = X_roi.shape[1]
rsa_scores = []

for r in range(n_regions):
    brain_dist = pdist(X_roi[:, r][:, None], metric='cityblock')
    RDM_brain = squareform(brain_dist)

    if np.all(np.isnan(brain_dist)) or np.std(brain_dist) == 0:
        rsa_scores.append((np.nan, np.nan))
        continue
    if np.std(fear_dist) == 0:
        rsa_scores.append((np.nan, np.nan))
        continue

    rho, pval = spearmanr(brain_dist, fear_dist)
    rsa_scores.append((rho, pval))
```

## Rank regions by RSA correlation

```
out = sorted([(labels[idx + 1], r, p) for idx, (r, p) in enumerate(rsa_scores)],
             key=lambda x: -x[1])

for name, rho, p in out[:10]:
    print(f"{name:<35}  ρ={rho:.3f}  p={p:.3g}")
```
| Brain Region                                 | ρ       | p        |
|---------------------------------------------|--------:|---------:|
| Superior Frontal Gyrus                      | 0.079   | 0.000887 |
| Precentral Gyrus                            | 0.037   | 0.121    |
| Postcentral Gyrus                           | 0.034   | 0.147    |
| Frontal Pole                                | 0.032   | 0.180    |
| Lateral Occipital Cortex, superior division | 0.025   | 0.287    |
| Middle Frontal Gyrus                        | 0.024   | 0.311    |
| Angular Gyrus                               | 0.018   | 0.437    |
| Cuneal Cortex                               | 0.016   | 0.508    |
| Paracingulate Gyrus                         | 0.014   | 0.561    |
| Parietal Opercular Cortex                   | 0.012   | 0.605    |

## Visualize the fear RDM and a brain RDMs

<iframe src="_static/rdm_slider_plot.html" width="100%" height="600px" frameborder="0"></iframe>

Now to try to interpret this:

The Superior Frontal Gyrus (SFG) shows the strongest (but small in absolute terms) and statistically significant correspondence with the fear rating RDM (ρ = 0.079, p ≈ 0.0009). This suggests that trial-wise fluctuations in subjective fear are "meaningfully" mirrored in this region's activity pattern. Since the SFG is implicated in executive function and emotion regulation, it could be counted as a plausible candidate for representing or modulating the conscious experience of fear. 

I'm not sure going all the way for the other regions is worthwhile due to their rather low correlations, but generally speaking, seeing some prefrontal and parietal assocition areas is a good sign.