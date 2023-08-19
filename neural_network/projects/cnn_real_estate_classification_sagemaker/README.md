# Real Estate Scene Classification

The goal of the project is to classify the images into one of 9 categories--- kitchens, bathrooms, backyard, etc. This helps with downstream tasks such as object detection, which is highly scene-dependent. For instance, we would want to detection features like granite countertops and stainless fridges in the kitchens, walk-in showers or bathtubs in the bathrooms, patios and grills in the backyards, and so on.

## Environment

To reproduce the development environment in SageMaker:

```
$ source activate tensorflow2_p310
# Or
$ source ~/anaconda3/etc/profile.d/conda.sh
$ conda activate tensorflow2_p310
$ pip install -r ./src/requirements.txt
``` 

## Documentation

Documentations for important API's are include in the `docs` directory.

## Structure of Source Directory

The src directory contains entry point scripts, utilities, and the `requirements.txt` required to successfully run the image classification project on AWS SageMaker. The src directory contains the following files:

```
src
├── __init__.py
├── base_trainer.py
├── config
│   ├── fine_tune
│   │   └── fine_tune.yaml
│   ├── vision_transformer
│   │   └── vision_transformer.yaml
│   └── main.yaml
├── custom_utils.py
├── fine_tune_entry.py
├── ingest_data.py
├── requirements.txt
└── vision_transformer_entry.py
```

* The `config` directory contains the `hydra` configuration yaml files. The typical configurations include:

    - AWS configurations: S3 bucket, region, framework version, etc.
    - Meta data for training: class labels, number of channels, image size, validation size, etc.
    - Other configurations for training: computing resources (instance types), spot instance set up, etc.
 
* The `ingest_data.py` script loads the raw data zip file from s3, reshapes the images, splits the data into train-val-test, and uploads the images as tensorflow dataset objects `tf.data.Dataset`. This part of the workflow is more ad-hoc and can benefit from more iterations and scoping as we move to better ways to store and access new images.

* The `vision_transformer_entry` and `fine_tune_entry.py` scripts are entry points that are used for SageMaker training jobs and hyperparameter jobs.

    - For the fine-tuning, the model is first initialized and trained at the top (dense layers) and then fine-tuned with more layers released (see training scripts for more details). We can choose from one of three state-of-the-art CNN architectures--- ResNet50v2, Xception, and VGG19, which is a hyperparameter. We enable three training modes--- local (CPU-based), single host multi-device (GPUs), and multiworker multi-device (GPUs).

    - For the vision-transformer, we implement an improved version of the architecture based on the paper [Vision Transformer for Small-Size Datasets](https://arxiv.org/abs/2112.13492v1).
 
    A note on the implementation is that all the modeling is carried out using tensorflow 2.12.0 at the time of writing this documentation. Since then, tensorflow 2.13.0 has become available as a SageMaker training image, and so we can swtich to take advantage of the new focal loss function, which is an improved loss function for handling class imbalance. This option can be toggled using the `use_focal_loss` hyperparameter.

* The `custom_utils.py` module contains utility functions for training and analysis.

When training begins, the files located in the `src` directory (including the `requirements.txt` file) will be copied onto the training docker image.

---

&nbsp;

## Results on Validation Data

<table id="T_3ed56_">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th class="col_heading level0 col0">precision</th>
      <th class="col_heading level0 col1">recall</th>
      <th class="col_heading level0 col2">specificity</th>
      <th class="col_heading level0 col3">f1</th>
      <th class="col_heading level0 col4">geometric_mean</th>
      <th class="col_heading level0 col5">index_balanced_accuracy</th>
      <th class="col_heading level0 col6">support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3ed56_level0_row0" class="row_heading level0 row0">backyard</th>
      <td id="T_3ed56_row0_col0" class="data row0 col0" style="background-color:yellow;">0.8190</td>
      <td id="T_3ed56_row0_col1" class="data row0 col1" style="background-color:green;">0.8829</td>
      <td id="T_3ed56_row0_col2" class="data row0 col2" style="background-color:green;">0.9747</td>
      <td id="T_3ed56_row0_col3" class="data row0 col3" style="background-color:green;">0.8498</td>
      <td id="T_3ed56_row0_col4" class="data row0 col4" style="background-color:green;">0.9277</td>
      <td id="T_3ed56_row0_col5" class="data row0 col5" style="background-color:green;">0.8527</td>
      <td id="T_3ed56_row0_col6" class="data row0 col6">205.0</td>
    </tr>
    <tr>
      <th id="T_3ed56_level0_row1" class="row_heading level0 row1">bathroom</th>
      <td id="T_3ed56_row1_col0" class="data row1 col0" style="background-color:green;">0.8617</td>
      <td id="T_3ed56_row1_col1" class="data row1 col1" style="background-color:green;">0.8898</td>
      <td id="T_3ed56_row1_col2" class="data row1 col2" style="background-color:green;">0.9772</td>
      <td id="T_3ed56_row1_col3" class="data row1 col3" style="background-color:green;">0.8755</td>
      <td id="T_3ed56_row1_col4" class="data row1 col4" style="background-color:green;">0.9325</td>
      <td id="T_3ed56_row1_col5" class="data row1 col5" style="background-color:green;">0.8619</td>
      <td id="T_3ed56_row1_col6" class="data row1 col6">245.0</td>
    </tr>
    <tr>
      <th id="T_3ed56_level0_row2" class="row_heading level0 row2">bedroom</th>
      <td id="T_3ed56_row2_col0" class="data row2 col0" style="background-color:yellow;">0.7885</td>
      <td id="T_3ed56_row2_col1" class="data row2 col1" style="background-color:yellow;">0.8200</td>
      <td id="T_3ed56_row2_col2" class="data row2 col2" style="background-color:green;">0.9364</td>
      <td id="T_3ed56_row2_col3" class="data row2 col3" style="background-color:yellow;">0.8039</td>
      <td id="T_3ed56_row2_col4" class="data row2 col4" style="background-color:green;">0.8763</td>
      <td id="T_3ed56_row2_col5" class="data row2 col5" style="background-color:yellow;">0.7589</td>
      <td id="T_3ed56_row2_col6" class="data row2 col6">400.0</td>
    </tr>
    <tr>
      <th id="T_3ed56_level0_row3" class="row_heading level0 row3">diningRoom</th>
      <td id="T_3ed56_row3_col0" class="data row3 col0" style="background-color:red;">0.4000</td>
      <td id="T_3ed56_row3_col1" class="data row3 col1" style="background-color:red;">0.1379</td>
      <td id="T_3ed56_row3_col2" class="data row3 col2" style="background-color:green;">0.9966</td>
      <td id="T_3ed56_row3_col3" class="data row3 col3" style="background-color:red;">0.2051</td>
      <td id="T_3ed56_row3_col4" class="data row3 col4" style="background-color:red;">0.3708</td>
      <td id="T_3ed56_row3_col5" class="data row3 col5" style="background-color:red;">0.1257</td>
      <td id="T_3ed56_row3_col6" class="data row3 col6">29.0</td>
    </tr>
    <tr>
      <th id="T_3ed56_level0_row4" class="row_heading level0 row4">frontyard</th>
      <td id="T_3ed56_row4_col0" class="data row4 col0" style="background-color:green;">0.8563</td>
      <td id="T_3ed56_row4_col1" class="data row4 col1" style="background-color:yellow;">0.8418</td>
      <td id="T_3ed56_row4_col2" class="data row4 col2" style="background-color:green;">0.9844</td>
      <td id="T_3ed56_row4_col3" class="data row4 col3" style="background-color:green;">0.8490</td>
      <td id="T_3ed56_row4_col4" class="data row4 col4" style="background-color:green;">0.9103</td>
      <td id="T_3ed56_row4_col5" class="data row4 col5" style="background-color:yellow;">0.8169</td>
      <td id="T_3ed56_row4_col6" class="data row4 col6">177.0</td>
    </tr>
    <tr>
      <th id="T_3ed56_level0_row5" class="row_heading level0 row5">hall</th>
      <td id="T_3ed56_row5_col0" class="data row5 col0" style="background-color:red;">0.4643</td>
      <td id="T_3ed56_row5_col1" class="data row5 col1" style="background-color:red;">0.4194</td>
      <td id="T_3ed56_row5_col2" class="data row5 col2" style="background-color:green;">0.9914</td>
      <td id="T_3ed56_row5_col3" class="data row5 col3" style="background-color:red;">0.4407</td>
      <td id="T_3ed56_row5_col4" class="data row5 col4" style="background-color:red;">0.6448</td>
      <td id="T_3ed56_row5_col5" class="data row5 col5" style="background-color:red;">0.3920</td>
      <td id="T_3ed56_row5_col6" class="data row5 col6">31.0</td>
    </tr>
    <tr>
      <th id="T_3ed56_level0_row6" class="row_heading level0 row6">kitchen</th>
      <td id="T_3ed56_row6_col0" class="data row6 col0" style="background-color:green;">0.8651</td>
      <td id="T_3ed56_row6_col1" class="data row6 col1" style="background-color:green;">0.8732</td>
      <td id="T_3ed56_row6_col2" class="data row6 col2" style="background-color:green;">0.9815</td>
      <td id="T_3ed56_row6_col3" class="data row6 col3" style="background-color:green;">0.8692</td>
      <td id="T_3ed56_row6_col4" class="data row6 col4" style="background-color:green;">0.9258</td>
      <td id="T_3ed56_row6_col5" class="data row6 col5" style="background-color:green;">0.8478</td>
      <td id="T_3ed56_row6_col6" class="data row6 col6">213.0</td>
    </tr>
    <tr>
      <th id="T_3ed56_level0_row7" class="row_heading level0 row7">livingRoom</th>
      <td id="T_3ed56_row7_col0" class="data row7 col0" style="background-color:yellow;">0.7708</td>
      <td id="T_3ed56_row7_col1" class="data row7 col1" style="background-color:red;">0.7416</td>
      <td id="T_3ed56_row7_col2" class="data row7 col2" style="background-color:green;">0.9258</td>
      <td id="T_3ed56_row7_col3" class="data row7 col3" style="background-color:yellow;">0.7560</td>
      <td id="T_3ed56_row7_col4" class="data row7 col4" style="background-color:yellow;">0.8286</td>
      <td id="T_3ed56_row7_col5" class="data row7 col5" style="background-color:red;">0.6740</td>
      <td id="T_3ed56_row7_col6" class="data row7 col6">449.0</td>
    </tr>
    <tr>
      <th id="T_3ed56_level0_row8" class="row_heading level0 row8">plan</th>
      <td id="T_3ed56_row8_col0" class="data row8 col0" style="background-color:green;">1.0000</td>
      <td id="T_3ed56_row8_col1" class="data row8 col1" style="background-color:green;">1.0000</td>
      <td id="T_3ed56_row8_col2" class="data row8 col2" style="background-color:green;">1.0000</td>
      <td id="T_3ed56_row8_col3" class="data row8 col3" style="background-color:green;">1.0000</td>
      <td id="T_3ed56_row8_col4" class="data row8 col4" style="background-color:green;">1.0000</td>
      <td id="T_3ed56_row8_col5" class="data row8 col5" style="background-color:green;">1.0000</td>
      <td id="T_3ed56_row8_col6" class="data row8 col6">34.0</td>
    </tr>
  </tbody>
</table>

&nbsp;

---

&nbsp;

## Results on Test Data

<table id="T_test">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th class="col_heading level0 col0">precision</th>
      <th class="col_heading level0 col1">recall</th>
      <th class="col_heading level0 col2">specificity</th>
      <th class="col_heading level0 col3">f1</th>
      <th class="col_heading level0 col4">geometric_mean</th>
      <th class="col_heading level0 col5">index_balanced_accuracy</th>
      <th class="col_heading level0 col6">support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_test_level0_row0" class="row_heading level0 row0">backyard</th>
      <td id="T_test_row0_col0" class="data row0 col0" style="background-color:yellow;">0.8273</td>
      <td id="T_test_row0_col1" class="data row0 col1" style="background-color:yellow;">0.8273</td>
      <td id="T_test_row0_col2" class="data row0 col2" style="background-color:green;">0.9697</td>
      <td id="T_test_row0_col3" class="data row0 col3" style="background-color:yellow;">0.8273</td>
      <td id="T_test_row0_col4" class="data row0 col4" style="background-color:green;">0.8957</td>
      <td id="T_test_row0_col5" class="data row0 col5" style="background-color:yellow;">0.7908</td>
      <td id="T_test_row0_col6" class="data row0 col6">139.0</td>
    </tr>
    <tr>
      <th id="T_test_level0_row1" class="row_heading level0 row1">bathroom</th>
      <td id="T_test_row1_col0" class="data row1 col0" style="background-color:yellow;">0.8188</td>
      <td id="T_test_row1_col1" class="data row1 col1" style="background-color:green;">0.8828</td>
      <td id="T_test_row1_col2" class="data row1 col2" style="background-color:green;">0.9688</td>
      <td id="T_test_row1_col3" class="data row1 col3" style="background-color:green;">0.8496</td>
      <td id="T_test_row1_col4" class="data row1 col4" style="background-color:green;">0.9248</td>
      <td id="T_test_row1_col5" class="data row1 col5" style="background-color:green;">0.8479</td>
      <td id="T_test_row1_col6" class="data row1 col6">128.0</td>
    </tr>
    <tr>
      <th id="T_test_level0_row2" class="row_heading level0 row2">bedroom</th>
      <td id="T_test_row2_col0" class="data row2 col0" style="background-color:yellow;">0.7702</td>
      <td id="T_test_row2_col1" class="data row2 col1" style="background-color:yellow;">0.8158</td>
      <td id="T_test_row2_col2" class="data row2 col2" style="background-color:green;">0.9524</td>
      <td id="T_test_row2_col3" class="data row2 col3" style="background-color:yellow;">0.7923</td>
      <td id="T_test_row2_col4" class="data row2 col4" style="background-color:green;">0.8815</td>
      <td id="T_test_row2_col5" class="data row2 col5" style="background-color:yellow;">0.7664</td>
      <td id="T_test_row2_col6" class="data row2 col6">152.0</td>
    </tr>
    <tr>
      <th id="T_test_level0_row3" class="row_heading level0 row3">diningRoom</th>
      <td id="T_test_row3_col0" class="data row3 col0" style="background-color:red;">0.6000</td>
      <td id="T_test_row3_col1" class="data row3 col1" style="background-color:red;">0.2000</td>
      <td id="T_test_row3_col2" class="data row3 col2" style="background-color:green;">0.9978</td>
      <td id="T_test_row3_col3" class="data row3 col3" style="background-color:red;">0.3000</td>
      <td id="T_test_row3_col4" class="data row3 col4" style="background-color:red;">0.4467</td>
      <td id="T_test_row3_col5" class="data row3 col5" style="background-color:red;">0.1836</td>
      <td id="T_test_row3_col6" class="data row3 col6">15.0</td>
    </tr>
    <tr>
      <th id="T_test_level0_row4" class="row_heading level0 row4">frontyard</th>
      <td id="T_test_row4_col0" class="data row4 col0" style="background-color:green;">0.8837</td>
      <td id="T_test_row4_col1" class="data row4 col1" style="background-color:yellow;">0.8444</td>
      <td id="T_test_row4_col2" class="data row4 col2" style="background-color:green;">0.9811</td>
      <td id="T_test_row4_col3" class="data row4 col3" style="background-color:green;">0.8636</td>
      <td id="T_test_row4_col4" class="data row4 col4" style="background-color:green;">0.9102</td>
      <td id="T_test_row4_col5" class="data row4 col5" style="background-color:yellow;">0.8172</td>
      <td id="T_test_row4_col6" class="data row4 col6">135.0</td>
    </tr>
    <tr>
      <th id="T_test_level0_row5" class="row_heading level0 row5">hall</th>
      <td id="T_test_row5_col0" class="data row5 col0" style="background-color:red;">0.4000</td>
      <td id="T_test_row5_col1" class="data row5 col1" style="background-color:red;">0.2000</td>
      <td id="T_test_row5_col2" class="data row5 col2" style="background-color:green;">0.9967</td>
      <td id="T_test_row5_col3" class="data row5 col3" style="background-color:red;">0.2667</td>
      <td id="T_test_row5_col4" class="data row5 col4" style="background-color:red;">0.4465</td>
      <td id="T_test_row5_col5" class="data row5 col5" style="background-color:red;">0.1835</td>
      <td id="T_test_row5_col6" class="data row5 col6">10.0</td>
    </tr>
    <tr>
      <th id="T_test_level0_row6" class="row_heading level0 row6">kitchen</th>
      <td id="T_test_row6_col0" class="data row6 col0" style="background-color:green;">0.9294</td>
      <td id="T_test_row6_col1" class="data row6 col1" style="background-color:green;">0.8541</td>
      <td id="T_test_row6_col2" class="data row6 col2" style="background-color:green;">0.9839</td>
      <td id="T_test_row6_col3" class="data row6 col3" style="background-color:green;">0.8901</td>
      <td id="T_test_row6_col4" class="data row6 col4" style="background-color:green;">0.9167</td>
      <td id="T_test_row6_col5" class="data row6 col5" style="background-color:yellow;">0.8294</td>
      <td id="T_test_row6_col6" class="data row6 col6">185.0</td>
    </tr>
    <tr>
      <th id="T_test_level0_row7" class="row_heading level0 row7">livingRoom</th>
      <td id="T_test_row7_col0" class="data row7 col0" style="background-color:red;">0.7024</td>
      <td id="T_test_row7_col1" class="data row7 col1" style="background-color:yellow;">0.7763</td>
      <td id="T_test_row7_col2" class="data row7 col2" style="background-color:green;">0.9357</td>
      <td id="T_test_row7_col3" class="data row7 col3" style="background-color:red;">0.7375</td>
      <td id="T_test_row7_col4" class="data row7 col4" style="background-color:green;">0.8523</td>
      <td id="T_test_row7_col5" class="data row7 col5" style="background-color:red;">0.7148</td>
      <td id="T_test_row7_col6" class="data row7 col6">152.0</td>
    </tr>
    <tr>
      <th id="T_test_level0_row8" class="row_heading level0 row8">plan</th>
      <td id="T_test_row8_col0" class="data row8 col0" style="background-color:green;">0.9333</td>
      <td id="T_test_row8_col1" class="data row8 col1" style="background-color:green;">1.0000</td>
      <td id="T_test_row8_col2" class="data row8 col2" style="background-color:green;">0.9989</td>
      <td id="T_test_row8_col3" class="data row8 col3" style="background-color:green;">0.9655</td>
      <td id="T_test_row8_col4" class="data row8 col4" style="background-color:green;">0.9995</td>
      <td id="T_test_row8_col5" class="data row8 col5" style="background-color:green;">0.9990</td>
      <td id="T_test_row8_col6" class="data row8 col6">14.0</td>
    </tr>
  </tbody>
</table>