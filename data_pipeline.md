# Data Pipeline
### Getting data
* CUB: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
* nabirds: https://dl.allaboutbirds.org/nabirds
* obd:
    * Get URLs to scrape from [this notebook](/notebooks/data/get_urls_oriental_birds.ipynb)
    * Download the images from obd from [this notebook](/notebooks/data/download_obd_images.ipynb)

* flikr: Scrape birds from flikr using [this tool]( https://github.com/antiboredom/flickr-scrape)

* Train a FasterRCNN [here](/notebooks/models/FasterRCNN.ipynb) to segment birds. Train on CUB/nabirds since they come with labels. Test on obd, flikr to produce bounding boxes
    * Trained model is available [here](https://drive.google.com/file/d/19L9fwZ-90EVhsibTD3agQFt4OfcTv8-h/view?usp=sharing)
### Processing the data
* Select images for modeling & process them in [this notebook](/notebooks/data/select_images_and_process.ipynb)
    * Subset each image to its bounding box
    * Select maximum modeling resolution (256x256)
        * Throw away images that are lower than that resolution, after preprocessing
    * Denoise the images a littlebit
    Save the results
One notebook:
* Build a good image classifier
    * Want to remove bad images. Defined in [this notebook](http://localhost:8888/notebooks/notebooks/data/model_for_good_data.ipynb)
* Save the images at various resolutions in [this notebook](http://localhost:8888/notebooks/notebooks/data/downsample_data.ipynb)
