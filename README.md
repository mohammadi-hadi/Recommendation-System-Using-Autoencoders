# Recommendation System Using Autoencoders

A movie recommender built with autoencoders on the MovieLens 1M dataset, following the approach of Ferreira et al. (2020).

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This study project implements a collaborative-filtering recommendation system based on autoencoders, inspired by the paper:

> Ferreira, D., Silva, S., Abelha, A., & Machado, J. (2020). Recommendation system using autoencoders. *Applied Sciences*, 10(16), 5510.

An autoencoder is trained to reconstruct each user's (sparse) rating vector; the reconstructed scores for unrated movies are then used as recommendation scores. Two architectures are compared, and the trained model is used to generate top-25 recommendations for a hand-crafted user profile.

## Data

[MovieLens 1M](https://grouplens.org/datasets/movielens/) (GroupLens): 1,000,209 ratings of 3,883 movies by 6,040 users, in the `movies.dat` / `users.dat` / `ratings.dat` format. The dataset is not included in this repository; download it from GroupLens.

## Methods

All in `Recommendation_System_Using_Autoencoders.ipynb` (written for Google Colab):

- Build a user-item rating matrix (6,040 x 3,952) with ratings scaled to (0, 1]; the raw matrix is about 95.8% sparse.
- Filter out movies with fewer than 20 ratings, leaving 3,043 items, and split users 70/30 into train and test sets.
- **Base model**: a single-hidden-layer autoencoder (256-unit bottleneck, tanh encoder / sigmoid decoder, He initialization), trained with Adam and MSE loss for 50 epochs.
- **Deep model**: a deeper encoder-decoder (512 -> 128 units with dropout), trained for 100 epochs at a lower learning rate.
- Evaluate by binarizing actual and reconstructed ratings at 0.7 (roughly "rated 4+") and computing the Dice matching coefficient, precision, and recall.
- Generate recommendations: feed a custom binary preference vector through the trained autoencoder and take the top-25 highest-scored movies.

## Results

Test-set scores from the executed notebook:

| Model | Dice coefficient | Precision | Recall |
|---|---|---|---|
| Base autoencoder (256 units) | 0.534 | 0.719 | 0.425 |
| Deep autoencoder (512-128, dropout) | 0.339 | 0.779 | 0.217 |

The shallow base model reconstructed relevant items better overall (test MSE about 0.014); the deeper model was more precise but recalled far fewer relevant movies.

## Repository structure

```
Recommendation-System-Using-Autoencoders/
├── Recommendation_System_Using_Autoencoders.ipynb   # Full analysis notebook (with outputs)
├── LICENSE
└── README.md
```

## Requirements / How to run

Python 3 with:

```
numpy pandas matplotlib scikit-learn tensorflow keras
```

1. Download the MovieLens 1M dataset and note the paths to `movies.dat`, `users.dat`, and `ratings.dat`.
2. Open the notebook (Colab or Jupyter) and update the dataset paths in the setup cells (the originals point to a mounted Google Drive).
3. Run the cells top to bottom.

## License

MIT — see [LICENSE](LICENSE).

## Author

Hadi Mohammadi — [mohammadi.cv](https://mohammadi.cv)
