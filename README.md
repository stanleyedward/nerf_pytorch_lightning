# Neural Radiance Fields
A neural radiance field ([NeRF](https://arxiv.org/abs/2003.08934)) is a method based on deep learning for reconstructing a three-dimensional representation of a scene from sparse two-dimensional images. The NeRF model enables learning of novel view synthesis, scene geometry, and the reflectance properties of the scene. 

1. [Setup](#Setup)
2. [Training](#Training)
3. [Evaluation](#Evaluation)
4. [References and Citations](#References-and-Citations)

<!--<div align="center">
  <img src="images/lego_16_epoch_400.gif" alt="Image">
</div>-->
<table>
  <tr>
    <td align="center"><img width="800" alt="nerf_output" src="images/lego_16_epoch_400.gif"><br>Novel Views</td>
    <td align="center"><img width="800" alt="sphere" src="images/speed_mesh.gif"><br>3D Reconstruction</td>
  </tr>
</table>

- Training: Loss and PSNR graphs

<table>
  <tr>
    <td align="center"><img width="800" alt="loss" src="images/loss_wandb_graph.png"><br>Loss</td>
    <td align="center"><img width="800" alt="psnr" src="images/psnr_wandb_graph.png"><br>PSNR</td>
  </tr>
</table>

- Device used: [TITAN X (Pascal) 250W / 12GB RAM](outputs/gpu.txt)

- Time to train: 12h 30m
 
- Training Dashboard: https://wandb.ai/stanleyedward/LegoNeRF/runs/h6yb8pnb/overview

- Testing Scores: 
    - Avg Loss: 0.0012457877128773586
    - Avg PSNR: 29.200356294523996

### Setup
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
- > Download Lego Dataset: https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a

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

### Training
`note:` [Setup](#Setup) should be complete

1. #### Change configurations 
    In the [config.py](config.py) file

    ```py
    """------------------------NeRF Config------------------------"""
    # data
    IMG_SIZE: int = 400
    BATCH_SIZE: int = 3072
    ...
    DEVICES: int = torch.cuda.device_count()
    MAX_EPOCHS: int = 17
    ```
2. #### Run the [train.py](train.py) script
    ```sh
    python train.py
    ```

### Evaluation
`note:` [Setup](#Setup) should be complete

1. #### Change configurations 
    In the [config.py](config.py) file

    ```py
    """------------------------NeRF Config------------------------"""
    ...
    #eval
    CKPT_DIR: str = "models/16_epoch_192_bins_400_nerf.ckpt" 
    CHUNK_SIZE: int = 20  # increase chunksize prevent CUDA out of memory errors
    OUTPUTS_DIR: str = "outputs" #folder you want to save the novel views in
    ```
2. #### Run the [eval.py](eval.py) script
    ```sh
    python eval.py
    ```


### References and Citations

- Neural radiance field. (2024, April 18). In Wikipedia. https://en.wikipedia.org/wiki/Neural_radiance_field
- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ArXiv. /abs/2003.08934

