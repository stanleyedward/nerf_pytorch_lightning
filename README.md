# Neural Radiance Fields
A neural radiance field (NeRF) is a method based on deep learning for reconstructing a three-dimensional representation of a scene from sparse two-dimensional images. The NeRF model enables learning of novel view synthesis, scene geometry, and the reflectance properties of the scene. 

1. [Dataset](#Dataset)
2. [Training](#Training)
3. [Evaluation](#Evaluation)
4. [Results](#Results)
5. [References and Citations](#References-and-Citations)

### Dataset
Lego Dataset: https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a

### Training
1. #### Clone and cd into the repository:

    ```sh
    git clone https://github.com/stanleyedward/nerf_pytorch_lightning.git
    cd nerf_pytorch_lightning
     ```

2. #### Create and activate the conda environment:

    ```sh
    conda env create -f environment.yaml
    conda activate nerf_pl
    ```

3. #### Add the dataset to the `dataset/` directory:
    ```sh
    dataset/
    ├── dataset_link.md
    └── lego
        ├── test
        ├── train
        ├── transforms_test.json
        ├── transforms_train.json
        ├── transforms_val.json
        └── val
    ```

4. #### Change configurations 
    In the [config.py](config.py) file
    ```py
    """------------------------NeRF Config------------------------"""
    # data
    DATA_DIR: str = "dataset/lego/"
    IMG_SIZE: int = 400
    BATCH_SIZE: int = 3072
    ...
    DEVICES: int = torch.cuda.device_count()
    MAX_EPOCHS: int = 17
    PRECISION: str = "32"
    ```
5. #### Run the [train.py](train.py) script
    ```sh
    python train.py
    ```

### Evaluation



### Results

- Run: https://wandb.ai/stanleyedward/LegoNeRF/runs/h6yb8pnb/overview
- Project: https://wandb.ai/stanleyedward/LegoNeRF/


### References and Citations

- Neural radiance field. (2024, April 18). In Wikipedia. https://en.wikipedia.org/wiki/Neural_radiance_field