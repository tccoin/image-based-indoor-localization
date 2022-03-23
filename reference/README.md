Import the conda envrionment.yaml environment.

DL+Geometric+Motion model  --Use 7scenes/TUM/[ISSAC]   
1.Download 7scenes/TUM dataset. Unzip them separately.  
2.[7scenes] Run 'tf_keras/data_preprocessing/Main_splitdata.py'. It convert the 7 scenes to standard format.  
2.[TUM] Run '\dev\python\data_preprocessing\Main_splitdata.m'.  
4.[ISSAC] Run '\dev\matlab\OrganizeIssacData.m'
5.Run '/dev/python/data_preprocessing/generate_tf_records_7scenes.py'. Set the path correctly at line 365  
6.Run '/dev/sizemese_zone/run_siamese.py'. .If you use other dataset, please search for keywork 'Modify here dataset_name'. Set the path correctly at line 830-832  
7.Run\dev\matlab\Main_TUM_1GL.m/Main_1GL.m This runs the geometric locator  

Task: rewrite Main_1GL.m as python


## Citation
* Song, Jingwei, Mitesh Patel, and Maani Ghaffari. "Fusing Convolutional Neural Network and Geometric Constraint for Image-based Indoor Localization." IEEE Robotics and Automation Letters (2022). https://arxiv.org/pdf/2201.01408.pdf 
```
@article{song2022fusing,
  title={Fusing Convolutional Neural Network and Geometric Constraint for Image-based Indoor Localization},
  author={Song, Jingwei and Patel, Mitesh and Ghaffari, Maani},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  pages={1-8},
  publisher={IEEE}
 }
```
