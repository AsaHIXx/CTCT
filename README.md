
## code for **Deep transfer learning enables lesion tracing of circulating tumor cells(CTC-Tracer)**


## Prerequisites

To install requirements:

```setup
pip install -r requirements.txt
```

- Python 3.8.0
- GPU Memory: 3GB+
- Pytorch 1.11.0
- Any Computer with a GPU

## DEMO
- simple demos can be found in **./Demo.ipynb**
- if you want to reproduce training process, you need to download the source dataset used in this study(sparse_50318_for_26types.npz) from the website(`http://117.25.169.110:1032/`).

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
Rscript ./utils/preprocess.R -i <file path>
```
and
```
python ./utils/preprocess.py --file <outputs from R> --save_path <path to save>
```
final outputs are the logTPM gene expression csv files.
## Training
**Train model** (transductive prediction model)
> fill the paramters in ./config/ctc_net.yaml and run 
```
python ctc_transductive_training.py
```
- model training step
- train log file/model checkpoint can be found in ./snapshot/<'PATH'>

## Evalution 
inductive prediction
> fill the paramters in ./config/val.yaml and run
```
python ctc_inductive_training.py
```
- return tabular prediction file at result file path(./results/<'PATH'>).

## Fine-tune
Fill the `init_weight` parameter in `config/ctc-net.yaml` and then fill the new data path. Then run `python ctc_transductive_training.py`. And a fine-tune training step starts.
## Visualization
Fill the parameters in `config/vis.yaml` and just run `python Visualization.py`. You can get a low-dimension embedding of the pretrained model.
## Gene Marker Finder
> Custom scripts for searching gene markers based on scanpy(1.9.1), see in ./utils/gene_finder{}.py
## License
This project is licensed under the MIT license.
- - -
