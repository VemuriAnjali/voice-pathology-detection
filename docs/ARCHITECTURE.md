# Architecture

```text
Local Dataset
     |
     v
Pre-processing
(LMS/DWT/normalization as represented in the project code)
     |
     v
Spectral Feature Generation
(LCST / Bispectrum feature stage)
     |
     v
MSRSpT-152 feature-learning stage
     |
     v
MultiObjWSA feature-fusion stage
     |
     v
Conv-HSA
Stage 1
     |
     v
MObjGB-Ensemble
Stage 2
 +----+---------+
 | XGBoost      |
 | LightGBM     |
 | CatBoost     |
 +--------------+
     |
     v
Voice Pathology Classification
```

The diagram summarizes the organization used by this repository. It should not be read as a claim that every mathematical operation is a line-for-line implementation of the published paper.
