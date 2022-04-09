# Image Based Indoor Localization

## Installation

1. Install libraries:
    ```
    conda create -n tf -c anaconda -c conda-forge tensorflow-gpu=2.4.1 opencv
    conda activate tf
    pip install progressbar nearpy tqdm h5py keras scipy matplotlib gtsam
    ```
2. Data processing
    1. Download [RGB-D Dataset 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and
       extract to `data` folder. Here we will use `chess` scene as example.
    2. Make sure you have following folder structure:
        ```
        ├─data
        │  ├─chess
        │  │  ├─seq-01
        │  │  ├─seq-02
        │  │  ├─seq-03
        │  │  ├─seq-04
        │  │  ├─seq-05
        │  │  ├─seq-06
        ```
    3. Run `mkdir -p model/chess/weights/ model/chess/training_data_info model/chess/cdf`
    4. Run `python scripts/split_dataset.py`
    5. Run `python scripts/generate_tf_records.py`
3. Compile fast searching module
    ```bash
    cd image_based_localization/search
    mkdir cmake-build-release
    cd cmake-build-release
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    ```
4. Train the model
    1. Install tensorflow-gpu, opencv and related libraries.
    1. Create folders for storing results:
        ```
        ├─model
        │  └─chess
        │      ├─training_data_info
        │      └─weights
        ```
    1. Run `python image_based_localization/siamese_network/train.py`. See all available arguments by
       running `python image_based_localization/siamese_network/train.py -h`.
    1. Run `python image_based_localization/siamese_network/test.py`. See all available arguments by
       running `python image_based_localization/siamese_network/test.py -h`.

## Project Info

week 1: Feb 21-Feb 27, week 7: Apr 4-Apr 10.

1. Run original codes and improve README for reference code
1. Read the dataset into Python
1. Work on improvements
    1. network/image search
    2. single image locator with NID
    3. Replace motion model with pose graph
1. Run tests & improve the documents
1. write report and slides

## Documents

- [Fusing Convolutional Neural Network and Geometric Constraint for Image-based Indoor Localization](https://arxiv.org/abs/2201.01408)
- [RGB-D Dataset 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Slides introducing the method](https://docs.google.com/presentation/d/1TcP9ghPcuDF08yf6W7LYyVBT8AwY06my/edit?usp=sharing&ouid=113322968888661125678&rtpof=true&sd=true)
