# Debiasing Facial Detection Systems 
## Motivation

In this project, I’m exploring two critical aspects of applied deep learning: facial detection and algorithmic bias.

As AI becomes more widely deployed, building fair and unbiased models is essential for ensuring trust and acceptance. 
Consider the seemingly straightforward task of facial detection: deciding whether an image contains a face. Despite its apparent simplicity, this task can exhibit 
systematic biases across different demographic groups.

To address this, I’m investigating a method inspired by [this](http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf) approach, which learns the *latent variables* of facial datasets and adaptively re-samples the training data, mitigating any biases that may be present in order to train a *debiased* model. The goal is to reduce biases and train a more balanced, debiased facial detection model.

## Dataset Overview
- Positive Examples: CelebA Dataset (over 200,000 celebrity face images)
- Negative Examples: ImageNet (varied non-human categories)
- Test Set: Balanced across four demographics, labeled by Fitzpatrick skin type and gender:
    - Light Female
    - Light Male
    - Dark Female
    - Dark Male

## Baseline: Standard CNN Classifier
A standard CNN architecture is trained to distinguish between "face" and "not face" images. The model consists of:

Multiple convolutional layers (feature extraction)

Batch normalization (training stability)

Two fully connected layers (classification)

## Performance:

High overall accuracy on the training set (e.g., 99.54%)

Observed Bias: When evaluated on the balanced test set, the model's accuracy varies significantly across demographic groups, typically favoring over-represented categories (e.g., light-skinned females).

## Why Does Bias Occur?
Imbalanced Training Data: Over-representation of certain features (e.g., light skin, female faces) in the training set leads to higher accuracy for those groups.

Manual Annotation Limitations: Annotating and balancing every possible subgroup is labor-intensive and may not capture all sources of bias.

## Unsupervised Debiasing with Variational Autoencoders (VAEs)
### Key Idea
Instead of manual annotation, use a VAE to learn the latent structure of the data in an unsupervised manner. The VAE encodes each image into a set of latent variables, capturing underlying features (e.g., skin tone, accessories, pose).

### VAE Loss Function
The loss combines:
- Latent Loss: Kullback-Leibler (KL) divergence between learned latent variables and a unit Gaussian.
- Reconstruction Loss: Measures how well the decoded image matches the original.

### Reparameterization Trick
Allows differentiable sampling from the latent space, enabling backpropagation during training.

Debiasing Variational Autoencoder (DB-VAE)
#### Architecture
- Encoder: Standard CNN classifier, outputs both class prediction and latent variables.
- Decoder: Reconstructs the input image from latent variables.
- Latent Dimension: Typically set to 100.

#### Loss Function
- For face images: Combines VAE loss and classification loss.
- For non-face images: Only classification loss is used.

#### Adaptive Resampling
After each epoch, the model:

- Estimates the frequency of each latent feature in the dataset.
- Increases sampling probability for images with rare features (e.g., dark skin, hats).
- Decreases sampling probability for over-represented features.
- This adaptive resampling ensures the model is exposed more frequently to under-represented features, reducing bias.

## Training and Evaluation
1. Train Baseline CNN: Observe demographic performance disparities.
2. Train DB-VAE: Use adaptive resampling based on learned latent variables.
3. Evaluate Both Models: Compare classification probabilities across the four demographic groups.


## Conclusion
Latent variable modeling with VAEs enables unsupervised discovery of under-represented features in facial datasets.

Adaptive resampling based on these latent features during training leads to debiased facial detection models.

The DB-VAE approach mitigates algorithmic b
