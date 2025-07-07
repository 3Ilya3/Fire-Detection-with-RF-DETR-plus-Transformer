Telegram: @wake_and_bake

# Dataset for RF-DETR must be in COCO-format:
```
dataset/
│
├── train/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── ...
│   └── _annotations.coco.json
│
├── val/
│   ├── img1.jpg
│   ├── ...
│   └── _annotations.coco.json 
│
└── test/          
    ├── img1.jpg
    ├── ...
    └── _annotations.coco.json
```

## Images from dataset:

![Fire](images_for_README/image6.jpg)
![Smoke](images_for_README/image7.jpg)
![Fire and smoke](images_for_README/image8.jpg)

# Training results
![Metrics plot](images_for_README/metrics_plot.png)
* map95: 0.572
* map50: 0.855

On the left is an image with annotations, on the right is the model's detection:
![Example of detection](images_for_README/image1.png)

![Example of detection](images_for_README/image2.png)

![Example of detection](images_for_README/image3.png)

The main issue was the model's false detection of objects with red light hues—such as emergency lights, headlights, streetlights, and other similar sources—as fire:

![Example of false detection](images_for_README/image4.png)

![Example of false detection](images_for_README/image5.png)

To address this,  I implemented a transformer-based module that analyzes temporal dynamics across frame sequences. This upgrade significantly improved the model’s ability to distinguish real fire from visual artifacts.

# Custom dataset for transformer:
```
dataset/
│
├── train/
│   ├── annotations.json
│   ├── ...
│   └──  seq_i/
│        ├── frame_000.jpg
│        ├── ...
│        └── frame_029.jpg
│
└── test/          
    ├── annotations.json
    ├── ...
    └── seq_i/
        ├── frame_000.jpg
        ├── ...
        └── frame_029.jpg
```

## Sequences from transformer dataset:
![Fire sequence](images_for_README/image9.png)

![Fire sequence](images_for_README/image10.png)

![Smoke sequence](images_for_README/image11.png)

![Lighting sequence](images_for_README/image12.png)

![Random sequence](images_for_README/image13.png)

# Training transformer results:
	* Test accuracy: 97,3%; 
	* Test loss: 0,036;
	* Precision: 0,941; 
	* Recall: 1; 
	* F1 score: 0,969.

# Results
![Fire detection](images_for_README/image14.png)

![Fire detection](images_for_README/image15.png)

![Smoke detection](images_for_README/image16.png)

![Smoke detection](images_for_README/image17.png)

![Lighting sequence](images_for_README/image18.png)

![Lighting sequence](images_for_README/image19.png)

