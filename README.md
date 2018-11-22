# Deep Virtual Try-on with Clothes Transform
Source code for paper "Deep Virtual Try-on with Clothes Transform"
<img height="300" src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/introduction.png">

## Overall Architecture
<img height="500" src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/All.png">

## Step1: CAGAN 
<img height="200" src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/CAGAN.png">

### code and data ###
* Training:  `CAGAN.py`
```
python CAGAN.py
```
* Testing: `Testing_with_fixed_data.py`
```
python Testing_with_fixed_data.py
```
* Data: `MVC_image_pairs_resize_new.zip`

### parameters in code ###

#### Training: CAGAN.py

* Data should be put in

`"./MVC_image_pairs_resize_new/1/*.jpg"` (for person images)

`"./MVC_image_pairs_resize_new/5/*.jpg"` (for clothes images)

> 470: data = "data folder name"
>
> 471: train_A = "person images folder name"
>
> 473: filenames_1 = "person images folder name"
>
> 474: filenames_5 = "clothes images folder name"
>
> 617, 618: set "save model path" 

#### Testing: Testing_with_fixed_data.py

* Data should be put in

`"./MVC_image_pairs_resize_new/1/*.jpg"` (for person images)

`"./MVC_image_pairs_resize_new/5_test/*.jpg"` (for clothes images)

>215: set "model path"
>
>220: data = "data folder name"
>
>221: train_A = "person images folder name"
>
>222: filenames_5 = "clothes images folder name"
>
>224: out_root_dir = "output folder name"
>
>225: origin_dir = "save input person images"
>
>226: target_dir = "save target clothes images"
>
>227: output_dir = "save output images"
>
>228: mask_dir = "save output masks"
>
>230: testing_number = "how much data you want to test"


## Step2: Segmentation ##
<img height="200" src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/segmentation.png">

### code ###
https://github.com/Engineering-Course/LIP_SSL

* Modify mask: `modify_mask.m`

* Save the masks file to png file: `show.m`

* Combine all the masks: `combine_with_CAGANmask.m`


## Step3: Transform ##
<img height="100" src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/warping.png">

### code and data ###

* Training: `unet.py` `data.py`
```
python unet.py
```

* Testing: `Testing_unet.py`
```
python Testing_unet.py
```
* Data: `transform_data.zip` `transform_test_data.zip`

### parameters in code ###
#### Training: unet.py

>336: model_dir = "save model path"
>
>337: result_dir = "save results path"
>
>223: set "loss type"

#### data.py

>15: set "data path"


#### Testing: Testing_unet.py

>16: test_data_path = "data path"
>
>17: test_img_folder = "target clothes image folder name"
>
>18: test_mask_folder = "mask folder name"
>
>19: model_name = "model name"
>
>20: result_dir = "save results path"


## Step4: Combination ##
<img height="200" src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/combine.png">

### code ###
`Combine_image.m`

## Results
<img src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/result1.png">
<img src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/result2.png">
<img src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/result3.png">
<img src="https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform/blob/master/readme_img/condition.png">
