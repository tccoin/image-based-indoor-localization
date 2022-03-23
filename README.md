# Image Based Indoor Localization

## Installation

1. Data processing
    1. Download [RGB-D Dataset 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and extract to `data` folder. Here we will use `chess` scene as example.
    1. Make sure you have following folder structure:
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
    1. Run `python scripts/split_dataset.py`
    1. Run `python scripts/generate_tf_records.py`

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
