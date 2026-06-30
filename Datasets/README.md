First, you need to download the dataset [\'gcjpy.zip, data.jsonl, function.json\'] from [link](https://1drv.ms/f/c/86f3e9013f5ba441/IgB1oQIRKBSSRa8-RUTuaCT7Aa_4iBhONx1ir71sl-WKKNY?e=zyaPmR).

# Authorship_Attribution

First, you need to download the dataset from [link](https://1drv.ms/f/c/86f3e9013f5ba441/IgB1oQIRKBSSRa8-RUTuaCT7Aa_4iBhONx1ir71sl-WKKNY?e=zyaPmR). Then, you need to decompress the `gcjpy.zip` file to the `Authorship_Attribution/CodeLM_name`. For example:

```
unzip gcjpy.zip

cd Authorship_Attribution/codebert-mlm
mv ../../gcjpy ./
```

Then, you can run the following command to preprocess the datasets. For example:

```
cd codebert-mlm
python process.py
python substitutes.py
python carrot_data_get.py
```

❕**Notes:** The labels of preprocessed dataset rely on the directory list of your machine, so it's possible that the data generated on your side is quite different from ours. You may need to fine-tune your model again.

## Fine-tune CodeLM

We use full train data for fine-tuning. The training cost is 10 mins on 4*P100-16G. We use full valid data to evaluate during training. For example:

```shell
cd CodeBert/Authorship_Attribution/saved_models
python fine-tuning.py
```


# Clone_detection

Then, you need to decompress the `data.jsonl` file to the `Clone_detection/CodeLM_name/data_folder`. For example:

```
cd Clone_detection/codebert-mlm/data_folder
mv ../../../data.jsonl ./
```

Then, you can run the following command to preprocess the datasets. For example:

```
cd codebert-mlm
python process.py
python split_testset.py
python substitutes.py
```

❕**Notes:** The labels of preprocessed dataset rely on the directory list of your machine, so it's possible that the data generated on your side is quite different from ours. You may need to fine-tune your model again.

## Fine-tune CodeLM

We only use 10% training data to fine-tune and 10% valid data to evaluate during training. The training cost is 3 hours on 8*P100-16G. For example:

```shell
cd CodeBert/Clone_detection/saved_models
python fine-tuning.py
```


# Defect_detection

Then, you need to decompress the `function.json` file to the `Defect_detection/CodeLM_name`. For example:

```
cd Defect_detection/codebert-mlm
mv ../../function.json ./
```

Then, you can run the following command to preprocess the datasets. For example:

```
cd codebert-mlm
python preprocess.py
python substitutes_Test.py
```

❕**Notes:** The labels of preprocessed dataset rely on the directory list of your machine, so it's possible that the data generated on your side is quite different from ours. You may need to fine-tune your model again.

## Fine-tune CodeLM

We use full train data for fine-tuning. The training cost is 50 mins on 8*P100-16G. We use full valid data to evaluate during training.

```shell
cd CodeBert/Defect_detection/saved_models
python fine-tuning.py
```