# Conclusion

The aim of this project was not to pin down a singular neural signature of fear, but to explore how different analytic methods—-mass univariate analysis, predictive modeling, and representational similarity analysis—-characterize the neural architecture of a psychologically rich construct. The results reflect the mixed promises and limits of each approach.

## Approach 1: Univariate

The mass univariate GLM yielded localized activation maps that were ~statistically robust and, broadly speaking, consistent with emotion-related regions. Notably, while individual contrasts revealed task-sensitive clusters, they did not provide an interpretable trial-level account of fear ratings. This reflects the GLM’s strength in identifying condition-level effects, but also its limitation in modeling graded psychological variables.

## Approach 2: Predictive

Predictive modeling, in contrast, offered trial-level predictions but failed to generalize meaningfully, with poor performance in classifying animal category and regressing fear ratings; as well as near-chance performance in classifying ordinal fear ratings. This suggests that the multivariate signal at the voxel level may be too noisy or spatially distributed for straightforward decoding in a limited data regime. It also highlights the fragility of inference-by-prediction when signal strength is weak or psychologically diffuse.

## Approach 3: Representational

Representational Similarity Analysis (RSA) offered a middle ground. It revealed that certain frontal regions—-particularly the superior frontal gyrus—-exhibit neural representational geometries that align modestly but significantly with behavioral fear structure across trials. This suggests that fear is not only processed in canonical limbic structures but may be represented, at least abstractly, in distributed cortical circuits involved in appraisal and regulation.

## Takehome

Taken together, these results do not point to a singular neural “signature” of fear, but rather show how fear becomes legible in different ways depending on the lens applied. The GLM captures reliable spatial contrasts, predictive modeling asks whether neural patterns are sufficient for inference, and RSA examines whether psychological similarity is mirrored in brain dynamics. Each method reveals partial truths.

If there is a takeaway, it is methodological rather than anatomical: capturing the neural instantiation of psychological constructs may require triangulation across models and perspectives. In the future, I'd like to see more papers that try all three of these approaches (including both decoding and encoding) and I hope to work on this a bit more, potentially with more of Darius' data once it's been collected, to try to lead by example. 

In the meantime however, I'm quite pleased with what I able to learn and experiment with over the course of the last four weeks of Brainhack School 2025. 

