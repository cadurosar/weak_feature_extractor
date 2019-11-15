# Weakly labeled sound feature extractor

Fork from the official github for the "Knowledge Transfer From Weakly Labeled Audio Using Convolutional Neural Network For Sound Events And Scenes" paper. https://github.com/anuragkr90/weak_feature_extractor

The only differences are that this is adapted to python3 and compressed to a single file to be easily imported. For citations always cite the original work https://arxiv.org/pdf/1711.01369.pdf

You can test this with 

```
python feature_extractor sample.wav
```

The output should be

```
(1, 1024) 3.4985018 0.0 0.70316136 0.7753519
```
