### Loss Functions/Regularization
* Wasserstein-gp
* Hinge
* Relativistic Hinge
* Logistic R1

* Regularization
    * R1
      * Scale penalty based on number of output features
      * Some hints that this being higher might be helpful
    * Spectral

### Archiecture Choices
* Latent Size

### Optimizer Choices
* G learning rate synthesis
* G learning rate mapping
* D learning rate

### Data Choices
* \# of epochs per size

### Normalization
* Does normalizing the input data help at all?
  * We do it for nomral cnns, why not for these cnns?
  * Easy to unnormalize the output of the generator

# Assumptions/Hopes
* Latent Size
  * I don't want to run experiments with latent size==512 b/c that's too big for my machine :(... takes a whle
  * Run with size==128 instead
  * After completing what I want of this table, will increase 512 to make sure still performs reasonably

# Training Schedule:
* 5 Epochs @ 4,8,16
* 10 @ 32
* 5 @ 64 for fadein
* 10 @ 64

# Results
FID End = FID after runnning the experiment I've described above
FID Extra = Does FID go down further after training up to FID End?

Default LRs:
* GMap: 1e-5
* GSynth: 1e-3
* Discrim: 1e-5

* NOTES:
  * For w-gp, hinge there was a bug in updating the mean latent for the truncation; was only using the first one, not all
| Loss FN  | GMap LR  | GSynth LR  | DisLR  | FID End | Fid Extra | Fid Extra \#Epochs | Notes | 
|:---|:---|:---|:---|:---|:---|:---| :---|
| w-gp |1e-5   |1e-3   |1e-3   | 23.09   | 21.74  | 10  |
| hinge |1e-5   |1e-3   |1e-3   | 23.57  | 21.7  | 10  |   |
| r_hinge |1e-5   |1e-3   |1e-3   |   |   |   |   |
| logistic_r1 $\gamma=...$ |1e-5   |1e-3   |1e-3   |   |   |   |   |
|   |   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |     |




Folder name structure:
[loss_function]\_[model_part_lr, if different than default]\_[regularization_and_amount, if different than default]