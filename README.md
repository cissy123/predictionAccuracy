# predictionAccuracy
This project is used to generate haplogroup prediction accuracy under six machine learning algorithms.

## Dependence

### step1. Prepare data set
prepare your data set which contains multi excel files. As shown in the 'dataset_example' folder, each file contains multi samples and each sample has multi dimensions.
It should be noted that there should be at lease TWO Haplogroup of each file.

### step2. Configure
Open file 'multi_prediction.py'. you can configure  In the last few lines of code.
'dataset_folder' ---- name of your data set of step 1
'pred_alg' ---- prediction algorithms you wanna try, six types are supported
'output_file_name' ---- whatever you like
'specialAlleleList' ---- special alleles which may contain more than one value
'k_fold_num' ---- the parameter applied in K FOLD CROSS VALIDATION method while prediction, this number should not less than the samples number

### step3. Run
```
python multi_prediction.py
```
