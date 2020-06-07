Emotion-detection
* The folder structure is of the form:  
  src:
  * data (folder)
  * data_pre_processing (folder)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)

## Content
* Bài viết sử dụng so sánh kết quả của các model:
1) Model sử dụng mạng CNN on master.
2) Model sử dụng mạng CNN paper
3) Model sử dụng data pre-processing for CNN paper

## Basic Usage
1) Model sử dụng mạng CNN on master.
- Được trình bày tại master branch
2) Model sử dụng mạng CNN paper:
* First, checkout Tung_part branch and clone the repository.
* If you want to train this model, use:  
```bash
cd src
python emotions.py --mode train
```

* If you want to view the predictions without training again, you can download the pre-trained model from [here](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view?usp=sharing) and then run:  
```bash
cd src
python emotions.py --mode display
```
* The folder structure được sử dụng   
  src:
  * data (folder)
  * data_pre_processing (folder)
  * `Train_CNN_PP.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `Model/CNN_Paper_weight.h5` (file)
  * 'Model/CNN_Paper_model.h5' (file)
  * `CNN_PP_Print_matrix.py` (file)
3) Model sử dụng data pre-processing for CNN paper 
Với crop face model:
  * B1: Xử lí ảnh: Crop-face:
```bash
cd src
cd Pre_processing
python Crop_face_datatrain.py
python Crop_face_datatest.py
```
  * B2: Training:
  - Quay lại folder src:
``` cd -```
  - Train:
```bash
Train_CNN_crop_face.py --mode train
```
* The folder structure is of the form:  
  Pre:
  * data_af_preprocessing/Cropped_facial_data(folder data)
  * `Pre_processing/Crop_face_datatrain.py`(file)
  * `Pre_processing/Crop_face_datatrain.py`(file)
  * `Pre_processing/haarcascade_frontalface_alt.xml`(file)
  Train:
  * `Train_CNN_crop_face.py` (file)
  Save:
  * `Model/CNN_crop_face_weight.h5` (file)

Với facial correction model:
  * B1: Xử lí ảnh: facial correction:
```bash
cd src
cd Pre_processing
python facial_correction_datatest.py
python facial_correction_datatrain.py
```
  * B2: Training:
  - Quay lại folder src:
``` cd -```
  - Train:
```bash
Train_CNN_Facial_correction.py --mode train
```
* The folder structure is of the form:  
  Pre:
  * data_af_preprocessing/Cropped_facial_data(folder data)
  * `Pre_processing/facial_correction_datatest.py`(file)
  * `Pre_processing/facial_correction_datatrain.py`(file)
  * `Pre_processing/shape_predictor_5_face_landmarks.dat`(file)
  * `Pre_processing/haarcascade_mcs_nose.xml`(file)
  * `Pre_processing/haarcascade_eye.xml`(file)
  * `Pre_processing/haarcascade_frontalface_default.xml`(file)
  Train:
  * `Train_CNN_Facial_correction.py` (file)
  Save:
  * `Model/CNN_Facial_correction_model.h5` (file)
  * `Model/CNN_Facial_correction_weight.h5` (file)

Với Illu face:
  * B1: Xử lí ảnh: facial correction:
```bash
cd src
cd Pre_processing
python Illu_model_datatrain.py
python Illu_model_datatest.py
```
  * B2: Training:
  - Quay lại folder src:
``` cd -```
  - Train:
```bash
Train_CNN_Facial_Illu.py --mode train
```
* The folder structure is of the form:  
  Pre:
  * data_af_preprocessing/Illu_data(folder data)
  Train:
  * `Train_CNN_Facial_Illu.py` (file)
  Save:
  * `'Model/CNN_Illu_weight.h5` (file)
  * `Model/CNN_Illu_model.h5` (file)

