# [OSIC Pulmonary Fibrosis Progression][competition]

## Context

The aim of this competition is to predict a patient’s severity of decline in lung function based 
on a CT scan of their lungs. Besides the scans there are also meta information, e.g. age, sex 
and smoking status (never smoked, ex-smoker, currently smokes).

In the dataset, you are provided with a baseline chest CT scan and associated clinical information 
for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up 
visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

The task is to predict the forced vital capacity (`FVC`) for each patient.


## Code Requirements

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 4 hours run-time
- External data, freely & publicly available, is allowed. This includes pre-trained models.


## Issues

1. The number of scans is vary from 30 to 200 for each user.
2. The format of the scans is Dicom (DCM).
3. Total amount of scans is around 34'000.
4. Although all scans were made at `Week = 0` there are samples in the train dataset that doesn't 
contain information about `FVC` at `Week = 0`
5. `FVC` measurements varies widely (the number of days between measurements is not constant).


## Decision Progress

Each patient has from 30 to 200 scan of their lungs. Also each of them has 9 measurements of `FVC`
in the `train.csv` dataset. Classical CV regression networks can't help here, because the size of
whole dataset is around 300'000 samples, which is too much for 4 hours run-time on GPU. So, I
decided to create an embedding network.

The idea is to take all images of the patient and convert it to a vector of numbers. This way, whole
dataset would still have only 1'500 samples. 

How can I do that?

#### First try (Prep.ipynb + NN.ipynb):

![][First try]

The embedding model takes a scan and some meta information. It returns a vector of 32 numbers. But 
how can I train this model?

My suggestion is to create another model based on the embedding one (like in transfer learning). And
the output of this model should be 5 most frequent measurements of `FVC` (at weeks 6, 8, 10, 12 
and 18)

__Prep.ipynb__ creates `train_base.csv` with mapping of FVCs and patients. And also it creates 
`.dat` files for each resized scan for `np.memmap` reading. `np.memmap` works faster then 
`pydicom.dcmread` and rising.  
__NN.ipynb__ trains and saves the embedding model.

But! The embedding model create vector of numbers for each scan. So, I still must create a vector of
number for all scans per patient. So, I decided to create a matrix of these numbers and take minimum 
and maximum. The final vector is consist of 64 values (32 minimums and 32 maximums)

The last step is regressor. It should take the vector and patient meta data and predict `FVC`. I 
decided to use `RandomForestRegressor`.  

__R2-Score__ of validation data is around 0.9. Although, the scoring in the competition is -11.

Current place: 1100/1600.

#### Second try

Or actually I should say second and later tries. 

I tried to tune embedding model with number of hidden layers or neurons. Also I tried different 
techniques to create `train_base.csv`, like setting weeks from -12 to 133, using `PolynomialFeatures`
and etc.

__R2-Score__ still 0.9. Competition score: OMG, -20.

Current place: too low, LMAO.

#### Third try

Okay, last embedding model produces the matrix of size n x 32 (n is number of scans of patient). And
I were creating the vector by simply taking minimum and maximum throw one axis. Maybe I should
create new embedding model that takes all images and produces vector of size 64.

So, the new model should handle multiple images. Instead of `Input(shape=[316, 316, 1])` I can use
`Input(shape=[None, 316, 316, 1])` But this case I must get rig of Depthwise layers, because they are
not implemented for 3D. The problem is `Conv3D` needs more memory. 

A-a-and yeah. OOM, baby! Even if I use only 1 patient, 100 images is too many.

__R2-Score__ Does not even exists. 


[competition]: https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/
[First try]: https://
