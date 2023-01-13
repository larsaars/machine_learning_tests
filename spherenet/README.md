# SpehereNet
## What is this?

Fits N-dimensional spheres under usage of a defined metric to a space as efficient as possible, everything that is in the given radius will be classified as belonging to a class.

## Usage

Attention: This classifier is extremly sensitive to outliers!

For tuning it might help, to remove these outliers (to filter). In certain cases (for example when having few data) it also helps to do a noisy data augmentation and reduce (`reduce_size()`) afterwards, because the given points will be used as centers.

The most important hyperparameter is the distance metric used.

A more detailled description will follow in the future.
