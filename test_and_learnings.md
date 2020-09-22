### Loss Functions/Regularization
* Wasserstein-gp
* Hinge
* Relativistic Hinge
* Logistic R1

* Regularization
    * R1
    * Spectral

### Archiecture Choices
* Latent Size

### Optimizer Choices
* G learning rate synthesis
* G learning rate mapping
* D learning rate

### Data Choices
* \# of epochs per size

# Assumptions/Hopes
* Latent Size
  * I don't want to run experiments with latent size==512 b/c that's too big for my machine :(... takes a whle
  * Run with size==128 instead
  * After completing what I want of this table, will increase 512 to make sure still performs reasonably

# TODOs:
1. Come up with a training schedule that will take approx 2 work days to run
   * Number of epochs for each resolution
   * Ideally:
     * First day is full 1...32 & full fade in of 64
     * Second day is continuing the training of 64

# Results
FID End = FID after runnning the experiment I've described above
FID Extra = Does FID go down further after training up to FID End?

Default LRs:
* GMap: 1e-5
* GSynth: 1e-3
* Discrim: 1e-5

| Loss FN  | GMap LR  | GSynth LR  | DLR  | FID End | Fid Extra | Notes | 
|:---|:---|:---|:---|:---|:---|:---|
| w-gp |1e-5   |1e-3   |1e-3   |   |   |   |
| hinge |1e-5   |1e-3   |1e-3   |   |   |   |
| r_hinge |1e-5   |1e-3   |1e-3   |   |   |   |
| logistic_r1 $\gamma=...$ |1e-5   |1e-3   |1e-3   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
Folder name structure:
[loss_function]_[model_part_lr, if different than default]