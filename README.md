# Dual Conditioned Motion Diffusion for Pose-Based Video Anomaly Detection

## Content
```
.
├── config
│   ├── Avenue
│   │   ├── dcmd_test.yaml
│   │   └── dcmd_train.yaml
│   ├── STC
│   │   ├── dcmd_test.yaml
│   │   └── dcmd_train.yaml
│   └── UBnormal
│       ├── dcmd_test.yaml
│       └── dcmd_train.yaml
├── environment.yaml
├── eval_DCMD.py
├── models
│   ├── common
│   │   └── components.py
│   ├── gcae
│   │   └── stsgcn.py
│   ├── dcmd.py
│   ├── transformer.py
│   ├── stsae
│   │    ├── stsae.py
│   │    └── stsae_unet.py
│   └── transformer.py
├── README.md
├── train_DCMD.py
└── utils
    ├── argparser.py
    ├── data.py
    ├── dataset.py
    ├── dataset_utils.py
    ├── diffusion_utils.py
    ├── ema.py
    ├── eval_utils.py
    ├── get_robust_data.py
    ├── __init__.py
    ├── model_utils.py
    ├── preprocessing.py
    └── tools.py
    
```

## Setup
### Environment
```sh
conda env create -f environment.yaml
conda activate dcmd
```

### Datasets
You can download the extracted poses for the datasets HR-Avenue, HR-ShanghaiTech and HR-UBnormal from the [GDRive](https://drive.google.com/drive/folders/1aUDiyi2FCc6nKTNuhMvpGG_zLZzMMc83?usp=drive_link).

Place the extracted folder in a `./data` folder and change the configs accordingly.


### **Training** 

To train DCMD:
```sh
python train_DCMD.py --config config/[Avenue/STC/UBnormal]/{config_name}.yaml
```


### Once trained, you can run the **Evaluation**

Test MoCoDAD
```sh
python eval_DCMD.py --config config/[Avenue/STC/UBnormal]/{config_name}.yaml
```
