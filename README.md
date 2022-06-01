
## code for **Circulating tumor cell annotation by deep transfer learning**


### Prerequisites

To install requirements:

```setup
pip install -r requirements.txt
```

- Python 3.8.0
- GPU Memory: 3GB+
- Pytorch 1.11.0



## Data preprocessing

input tabular data structure: 
```
      Barcode1 Barcode2 Barcode3 Barcode4 Barcode5
Gene1 ...
Gene2          ...     
Gene3                   ...
Gene4                            ...
Gene5                                     ...
```
> Gene symbol or ensemble ID is supported

Run data preprocess scripts(special file path and gene ID):
```
Rscript ./utils/preprocess.R
```
and
```
python ./utils/preprocess.py --file <outputs from R> --save_path <path to save>
```
final outputs are the logTPM gene expression csv files.
## Training
Train model (transductive prediction model)
> fill the paramters in ./config/ctc_net.yaml
```
python ctc_transductive_training.py
```
evalution (inductive prediction)
> fill the paramters in ./config/val.yaml
```
python ctc_inductive_training.py
```
> return tabular prediction file at result file path.

> source data used in this article can download from here.
