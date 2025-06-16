# Approach 2 - Predictive

Machine learning is usually used to determine the degree to which a set of variables, often called features, is predictive of a given outcome or target. When the features are brain stuff and the target is task- or stimulus-related stuff, it's called *decoding*. Conversely, when the target is the brain and the features are other stuff, that's *encoding*. But I can never really remember which is which on the fly. 

So before jumping into the analysis, I'd like to take the opportunity to go on a bit of a tangent--*or maybe more of a prologue since we haven't started yet*--to try to clear things up, mostly for myself. It may nevertheless be at least somewhat informative if you don't already have a firm grasp on it. 

---
## Encoding VS Decoding

The little story I tell myself is:

```{hint}
*Succesfully **decoding** a pattern of neural activity means predicting what it's about; solving the puzzle using a secret key (predictive modeling).* 

***Encoding** is about the kind of stuff in the world that is predictive of the neural activity of a brain.* 
```

But this little story still doesn't do much to convince me that the difference is all that meaningful. If it works one way, shoudn't it work both ways? What would it mean if not? To gain a better intuition about this, let's try to reason our way through the following 2 x 2 matrix:


|                                         | **$BRAIN$ Predicts $STUFF$** (GOOD Decoding)                                                     | **$BRAIN$ Does Not Predict $STUFF$** (BAD Decoding)                          |
| --------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **$STUFF$ Predicts $BRAIN$** (GOOD Encoding)         | **Bidirectional Predictability**<br> (Intuitive link between task/stimulus and brain response) | **Partial Encoding**<br> (Limited inference of $X$ from $Y$)     |
| **$STUFF$ Does Not Predict $BRAIN$** (BAD Encoding) | **Redundant/Confounding Patterns**<br> (Alternative factors drive $BRAIN$)              | **Complex or Nonlinear Mapping**<br> (Low mutual predictability) |

**1. $STUFF$ predicts $BRAIN$ *AND* $BRAIN$ predicts $STUFF$: Bidirectional Predictability**<br>
This is an ideal scenario which points to a straightforward relationship between input and output. Simple sensory mappings, like specific sound frequencies predictably activating specific neurons in the auditory cortex, where decoding these neurons can accurately reveal the frequency heard. Encoding explains the sensory representation, while decoding can reconstruct or predict stimuli based on neural patterns.

**2. $STUFF$ predicts $BRAIN$ *BUT* $BRAIN$ does not predict $STUFF$: Partial Encoding**<br>
This scenario means that the brain's activity may encode some general features of $STUFF$, but not in a way that's directly reversible. For example, emotionally intense scenes in a film reliably activate certain networks, yet the specific scene or emotion can’t be decoded from brain activity alone. It may hint at something like: *"Whatever computations the brain is doing are sensitive to $STUFF$, but probably use $STUFF$ in more complex ways than either the structure of $STUFF$ or the model relating $STUFF$ to $BRAIN$ can account for."*

**3. $STUFF$ does not predict $BRAIN$ *BUT* $BRAIN$ predicts $STUFF$: Redundant/Confounding Patterns**<br>
To me, this is the most counterintuitive quadrant. How can you predict the stimulus from brain activity if the stimulus doesn’t reliably cause the activity? It may be that brain activity co-varies with some latent variable that itself co-varies with $STUFF$. So decoding works, not because the brain is encoding $STUFF$ per se, but because the system has learned or internalized regularities that happen to align with $STUFF$.


**4. $STUFF$ does not predict $BRAIN$ *AND* $BRAIN$ does not predict $STUFF$: Complex or Nonlinear Mapping / Epistemic Blind Spot**<br>
This configuration reflects a breakdown in both encoding and decoding, suggesting a mismatch between what we measure and how the brain actually organizes information. Spontaneous thought is a good example: internal experience varies richly, but eludes both prediction from and inference via brain signals. Models fail not just due to noise, but possibly due to deep representational mismatch. This quadrant reminds us that some mappings may be nonlinear, latent, or fundamentally underconstrained given current tools.


---

So no, if it works one way it does not de facto work the other way and the distinction _is_ important in the grander scheme of things when one contends with the __kinds of claims that are licensed by one approach or the other__. In other words, what kinds of questions can each approach reasonably expect to be able to answer?

```{important}
**You're asking mechanistic questions**: *“What does this neuron/region respond to?”* → ***Encoding***

**You're asking inference questions**: *“Can I tell what the subject is seeing or intending from their brain activity?”* → ***Decoding***
```

In sum, both approaches can hint at what information is represented in the brain, but one allows for theoretical/mechanistic hypothesization (encoding) whereas the other does not (or so the story goes).

> *If our goal is merely to demonstrate that a brain region contains information about the experimental conditions, then the direction the model should operate in is a technical issue: One direction may be more convenient for capturing the relevant statistical dependencies (e.g. noise correlations among responses), but a model operating in either direction could support the inference that particular information is present in the code. If our goal is to test computational theories, however, then the direction that the model operates in matters, because it determines whether the model can be interpreted as a brain-computational model.<br>- Kriegeskorte & Douglas, 2019*

## Enough rambling

Let's get down and dirty. 

A lot of the setup for instantiating a machine learning model and running it on neural data is similar to the last step, so I'll pick up where the differences begin. The first thing is we need the structure the `brain_data` a bit differently to be able to feed it to a machine learning model.

There is no "in principle" correct way to structure/choose features in machine learning, and even less so in neuroscientific machine learning. A common approach, which we'll reproduce here, is to use flattened beta maps as features. In other words, the outputs of a first level GLM are used as features. As an FYI, some recent work has tried (and succeeded to varying degrees) using features that are "closer" to the original neural signal (i.e. taking the average of the HRF a couple seconds post stimulus presentation). But we'll stick to the basics for now. 

### Imports
```
import pandas as pd
import numpy as np
from nilearn import image, masking
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_epi_mask, intersect_masks
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_brain_mask
```

### Load data and fit FirstLevelModel
_Here, we could in principle use the exact same model as the last step, but for posterity's sake I'm including this step here again._

```
fmri_paths  = [f"ManyFaces/derivatives/sub-09/func/sub-09_task-videos_run-{r}_preproc_space-MNI_bold.nii.gz"
               for r in (1, 2, 3)]
event_paths = [f"ManyFaces/events/sub-09_ses-01_task-video_run-{r}_events.tsv"
               for r in (1, 2, 3)]

betas, y_cat, y_fear, run_id = [], [], [], []

for run_idx, (func, ev_path) in enumerate(zip(fmri_paths, event_paths), start=1):
    fmri_img   = image.load_img(func)
    events_raw = pd.read_csv(ev_path, sep="\t")
    events_raw = events_raw.rename(columns={"category": "trial_type"}) 
    events_raw["modulation"] = events_raw["rating"] - events_raw["rating"].mean() # Mean centered, twas recommended somewhere

    glm = FirstLevelModel(t_r=0.867, 
                          hrf_model="spm",
                          drift_model="cosine",
                          noise_model='ar1',
                          standardize=True,
                          high_pass=0.01,)
    glm.fit(fmri_img, events_raw)

    design = glm.design_matrices_[0] 
    trial_cols = design.columns.drop(["constant"] +  
                                     [c for c in design.columns if c.startswith("drift")])

    # beta maps for every trial
    for col in trial_cols:
        beta_img = glm.compute_contrast(col, output_type="effect_size")
        betas.append(beta_img)

    trial_events = events_raw.iloc[:len(trial_cols)]
    y_cat.extend(trial_events["trial_type"])
    y_fear.extend(trial_events["rating"])
    run_id.extend([run_idx] * len(trial_cols))
```

### Prepare a mask
Using a brain mask is necessary to extract only the meaningful, non-background voxels from each 3D beta image. This reduces each beta map to a 1D vector of relevant values, making the data compatible with machine learning models that expect structured, fixed-size feature arrays.

```
ref_img = image.load_img(fmri_paths[0]) 
mni_mask = load_mni152_brain_mask()
resampled_mask = resample_to_img(mni_mask, ref_img, interpolation='nearest')

masker = NiftiMasker(
    mask_img=resampled_mask,
    detrend=True,
    standardize="zscore_sample",
    low_pass=None,
    high_pass=0.008,
    t_r=0.867
)

# fit masker on all beta maps, then transform
masker.fit(betas)
X = masker.transform(betas) # shape (n_trials, n_voxels)
y_cat = np.array(y_cat)
y_fear = np.array(y_fear)
groups = np.array(run_id)
```

### Run our classification and regression
Oops, I kind of forgot to tell you, but we're actually building and testing _two_ models: one to predict (classify) the animal category and one to predict (continuously, so regression) the fear ratings. Why? Because why not. 

*I'm leaving hyperparameter tuning and fancy cross-validation aside for the moment so I can stay closer to the "neuro" side of this than the "data science" side, but will definitely pick back up on it in the future.*

```
logo = LeaveOneGroupOut()

# Category classification
clf = LogisticRegression(max_iter=500, multi_class="multinomial", C=1.0)
cat_scores = []

for train, test in logo.split(X, y_cat, groups):
    clf.fit(X[train], y_cat[train])
    preds = clf.predict(X[test])
    cat_scores.append(accuracy_score(y_cat[test], preds))

print("Category accuracy per left-out run:", cat_scores)
print("Mean accuracy:", np.mean(cat_scores))

# Fear-rating regression (ridge)
reg = Ridge(alpha=1.0)
fear_scores = []

for train, test in logo.split(X, y_fear, groups):
    reg.fit(X[train], y_fear[train])
    preds = reg.predict(X[test])

    """
    Change scoring metric for fear regression
    /alternatively, maybe switch from regression to multiclass classification
    TBD depending on the sparsity of the ratings
    """


    fear_scores.append(r2_score(y_fear[test], preds))

print("Fear rating R² per left-out run:", fear_scores)
print("Mean R²:", np.mean(fear_scores))
```
> *Output:*<br>Category accuracy per left-out run: [0.05, 0.15, 0.0]<br>Mean accuracy: 0.06666666666666667<br>Fear rating R² per left-out run: [-36.108291534089524, -234.85262631972967, -3.0725813057054214]<br>Mean R²: -91.34449971984152

These results are frankly *terrible*, but it is to be expected as machine learning is better suited to much **much** larger datasets.

Another thing that came to mind was to classify the fear ratings, as they are ordinal rather than purely continuous. 

```
fear_clf = LogisticRegression(max_iter=500, C=1.0)
fear_scores = []
for train, test in logo.split(X, y_fear, groups):
    fear_clf.fit(X[train], y_fear[train])
    preds = fear_clf.predict(X[test])
    fear_scores.append(accuracy_score(y_fear[test], preds)) 

print("Fear rating classification accuracy per left-out run:", fear_scores)
print("Mean accuracy:", np.mean(fear_scores))
```
> Fear rating classification accuracy per left-out run: [0.15, 0.35, 0.2]<br>
Mean accuracy: 0.2333333333333333

This is a bit better, and since we have 5 levels of fear, chance performance woul dbe around 20%. However, due to some non-negligeable class imbalance, these results are still pretty lackluster. 

```
unique, counts = np.unique(y_fear, return_counts=True)
print(np.asarray((unique, counts)).T)
```
> [[ 0. 23.]<br>
 [ 1. 15.]<br>
 [ 2.  8.]<br>
 [ 3.  4.]<br>
 [ 4. 10.]]

In an ideal scenario, we'd have good predictive ability and would then plot the beta maps against an anatomical brain and basically do the same thing we did in Approach 1, that is to try to figure out if the betas land in brain regions canonically associated with fear. If I return to these anlayses and get something better out of them, you can expect to see a map under this paragraph, but for now we'll move on to the last approach, RSA. 

## References
1. Kriegeskorte, N., & Douglas, P. K. (2019). *Interpreting encoding and decoding models.* **Current opinion in neurobiology, 55, 167-179.**