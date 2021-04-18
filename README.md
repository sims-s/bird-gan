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

The processed datasets at various resolutions are here: [4](https://drive.google.com/file/d/1rhP4wGYOKglsE4mmtPl4BjuEwzUqWv4h/view?usp=sharing), [8](https://drive.google.com/file/d/1aMyPcATkTKKGlBu1xQg-xtfXdaFbfozA/view?usp=sharing), [16](https://drive.google.com/file/d/113oLqOttOIZExJKJzLux_WmBzjjH3n7x/view?usp=sharing), [32](https://drive.google.com/file/d/1JRRWivjufzgwBST-gXzMolUXyKI4RFOV/view?usp=sharing), [64](https://drive.google.com/file/d/17L1NND_IRI978iSixvJumlbDHmby3Oeb/view?usp=sharing), [128](https://drive.google.com/file/d/1LnWqPjyLTX5gYHS8PrUommEmsqQSiVS0/view?usp=sharing), [256](https://drive.google.com/file/d/12rtNWwhxC-KSD0IEONLeTPIHvSyQj4dm/view?usp=sharing)

#### Sources
* Progan inspiredby: https://github.com/akanimax/pro_gan_pytorch/tree/master/pro_gan_pytorch
* Stylegan inspired by: https://github.com/huangzh13/StyleGAN.pytorch

