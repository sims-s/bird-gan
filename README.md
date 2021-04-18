# bird-gan
## Train ProGAN & StyleGAN to generate images of birds :bird:

#### Data
#### 4 Data Sources:
1. CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. NA Birds: https://dl.allaboutbirds.org/nabirds
3. Oriental Birds Database (OBD): http://orientalbirdimages.org/birdimages.php (See [these](https://github.com/sims-s/bird-gan/blob/master/notebooks/data/get_urls_oriental_birds.ipynb) [notebooks](https://github.com/sims-s/bird-gan/blob/master/notebooks/data/download_obd_images.ipynb) for downloading the data)
4. Flikr: Used [this repo](https://github.com/antiboredom/flickr-scrape) to scrape images from flikr.  
  
  
#### Data Processing:
1. Using bounding box labels from CUB/NA Birds, finetune a FasterRCNN in [this notebook](https://github.com/sims-s/bird-gan/blob/master/notebooks/models/FasterRCNN.ipynb)
2. Apply FasterRCNN to OBD and Flikr images to segment to just birds
3. (For all 4 data sources), choose a base resolution of 256. Throw away images that are lower resolution b/c there werne't many of them and dont want model to learn from "not real" images.
4. Want the images to not have a ton of diversity because that could be confusing for the model, so train a handfull of "filter models" in [this notebook](https://github.com/sims-s/bird-gan/blob/master/notebooks/data/model_for_good_data.ipynb) (More details @ the top of the notebook). These filter models are used on all datasets to remove images that are "nonstandard" in some way. 


#### Sources
* Progan inspiredby: https://github.com/akanimax/pro_gan_pytorch/tree/master/pro_gan_pytorch
* Stylegan inspired by: https://github.com/huangzh13/StyleGAN.pytorch

