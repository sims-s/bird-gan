# bird-gan
## Train ProGAN & StyleGAN to generate images of birds :bird:

### Data
#### 4 Data Sources:
1. CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. NA Birds: https://dl.allaboutbirds.org/nabirds
3. Oriental Birds Database (OBD): http://orientalbirdimages.org/birdimages.php (See [these](https://github.com/sims-s/bird-gan/blob/master/notebooks/data/get_urls_oriental_birds.ipynb) [notebooks](https://github.com/sims-s/bird-gan/blob/master/notebooks/data/download_obd_images.ipynb) for downloading the data).
4. Flikr: Used [this repo](https://github.com/antiboredom/flickr-scrape) to scrape images from flikr.  
  
  
### Data Processing:
1. Using bounding box labels from CUB/NA Birds, finetune a FasterRCNN in [this notebook](https://github.com/sims-s/bird-gan/blob/master/notebooks/models/FasterRCNN.ipynb)
2. Apply FasterRCNN to OBD and Flikr images to segment to just birds
3. (For all 4 data sources), choose a base resolution of 256. Throw away images that are lower resolution b/c there werne't many of them and dont want model to learn from "not real" images.
4. Want the images to not have a ton of diversity because that could be confusing for the model, so train a handfull of "filter models" in [this notebook](https://github.com/sims-s/bird-gan/blob/master/notebooks/data/model_for_good_data.ipynb) (More details @ the top of the notebook). These filter models are used on all datasets to remove images that are "nonstandard" in some way. 

### Results
#### Samples
![](https://github.com/sims-s/bird-gan/blob/master/results/NiceSamples.png)  
These samples are all cherrypicked becasue they look good. You can find some random samples [here](https://github.com/sims-s/bird-gan/blob/master/results/RandomSamples.png)


#### FID
FID was computed using 10,000 generated samples. The best model has a FID of 8.669. Here's a plot of FID during training:
![](https://github.com/sims-s/bird-gan/blob/master/results/FID.png)

#### Fixed Noise GIF
Take some fixed noise, and have all model checkpoints produce images using it. Plot results to see how model evolves during training. You can see more examples [here](https://github.com/sims-s/bird-gan/tree/master/results/fixed_noise_gifs)
![](https://github.com/sims-s/bird-gan/blob/master/results/fixed_noise_gifs/46.gif)

#### Style Mixing
Experimenting with mixing the styles at various reslutions. The images the first row & first column are the original images.
Then the i-jth entry of the grid has the original styles of the row's image, except at the specified resolutions where it has the style of the column's image.

![](https://github.com/sims-s/bird-gan/blob/master/results/style_mixing/StyleMix48.png)
![](https://github.com/sims-s/bird-gan/blob/master/results/style_mixing/StyleMix1632.png)
![](https://github.com/sims-s/bird-gan/blob/master/results/style_mixing/StyleMix64128.png)

#### Latent Interpolation
![](https://github.com/sims-s/bird-gan/blob/master/results/latent_interpolation/noise_vector_interpolation.gif)

#### Injected Noise Experiment
In addition to the noise input, noise is injected to the generator (Normal 0,1).  We can scale that noise to be as large or small as we'd like. Images in [this folder](https://github.com/sims-s/bird-gan/tree/master/results/per_channel_noise) are my experimentation with that. The suffix for each name is the scaling times 100 (e.g. scaling of 25 --> scale by .25). You can see with no noise the "birds" are very flat, but as we increase the noise more, they get taller again. Interesting.

### Sources
* Stylegan: https://arxiv.org/abs/1812.04948
* Progan: https://arxiv.org/abs/1710.10196
* Progan inspiredby: https://github.com/akanimax/pro_gan_pytorch/tree/master/pro_gan_pytorch
* Stylegan inspired by: https://github.com/huangzh13/StyleGAN.pytorch

