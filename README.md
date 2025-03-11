# Dual Conditioned Motion Diffusion for Pose-Based Video Anomaly Detection

## Abstract
Video Anomaly Detection (VAD) is essential for computer vision research. Existing VAD methods utilize either reconstruction-based or prediction-based frameworks. The former excels at detecting irregular patterns or structures, whereas the latter is capable of spotting abnormal deviations or trends. We address pose-based video anomaly detection and introduce a novel framework called Dual Conditioned Motion Diffusion (DCMD), which enjoys the advantages of both approaches. The DCMD integrates conditioned motion and conditioned embedding to comprehensively utilize the pose characteristics and latent semantics of observed movements, respectively. In the reverse diffusion process, a motion transformer is proposed to capture potential correlations from multi-layered characteristics within the spectrum space of human motion. To enhance the discriminability between normal and abnormal instances, we design a novel United Association Discrepancy (UAD) regularization that primarily relies on a Gaussian kernel-based time association and a self-attention-based global association. Finally, a mask completion strategy is introduced during the inference stage of the reverse diffusion process to enhance the utilization of conditioned motion for the prediction branch of anomaly detection. Extensive experiments conducted on four datasets demonstrate that our method dramatically outperforms state-of-the-art methods and exhibits superior generalization performance.
![image](https://github.com/guijiejie/DCMD-main/blob/main/architecture.png)
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

The checkpoints for the pretrained models can be found [GDRive](https://drive.google.com/drive/folders/1vvJv66-hi9D7DJKZGlzQzRhN0aYcwP05).

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{dualcondition25,
  title={Dual Conditioned Motion Diffusion for Pose-Based Video Anomaly Detection},
  author={Hongsong Wang, Andi Xu, Pinle Ding and Jie Gui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
</code></pre>
  </div>
</section>
