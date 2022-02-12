# precomputed_annotations

Make precomputed layers from a list of annotations. Currently only works for point annotations and does not include annotation properties. The specification for precomputed annotation layers is described here: https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md

Currently, the downsampling scheme is hardcoded: https://github.com/BrainCOGS/precomputed_annotations/blob/main/precomputed_annotations.py#L28

One would like to downsample using factors that make the volume maximally isotropic. The code in calculate_factors.py does that, but it needs to be integrated into the main code. 
