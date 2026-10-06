# PalmBridge: A Plug-and-Play Feature Alignment Framework for Open-Set Palmprint Verification

#### Abstract
> Palmprint recognition is widely used in biometric systems, yet real-world performance often degrades due to feature distribution shifts caused by heterogeneous deployment conditions. Most deep palmprint models assume a closed and stationary distribution, leading to overfitting to dataset-specific textures rather than learning domain-invariant representations. Although data augmentation is commonly used to mitigate this issue, it assumes augmented samples can approximate the target deployment distribution, an assumption that often fails under significant domain mismatch. To address this limitation, we propose PalmBridge, a plug-and-play feature-space alignment framework for open-set palmprint verification based on vector quantization. Rather than relying solely on data-level augmentation, PalmBridge learns a compact set of representative vectors directly from training features. During enrollment and verification, each feature vector is mapped to its nearest representative vector under a minimum-distance criterion, and the mapped vector is then blended with the original vector. This design suppresses nuisance variation induced by domain shifts while retaining discriminative identity cues. The representative vectors are jointly optimized with the backbone network using task supervision, a feature-consistency objective, and an orthogonality regularization term to form a stable and well-structured shared embedding space. Furthermore, we analyze feature-to-representative mappings via assignment consistency and collision rate to assess model's sensitivity to blending weights. Experiments on multiple palmprint datasets and backbone architectures show that PalmBridge consistently reduces EER in intra-dataset open-set evaluation and improves cross-dataset generalization with negligible to modest runtime overhead.


#### Citation
If our work is valuable to you, please cite our work:
```

```

#### Requirements

If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

#### Data Preprocessing
To help readers to reproduce our method, we also release our training, testing, gallery and query lists (including PolyU, Tongji, IITD, Multi-Spectrum datasets). If you wanna try our method in other datasets, you need to generate training and testing texts as follows:

```
python ./data/genText.py
```

#### Training
After you prepare the training, testing, gallery and query texts, then you can directly run our training code as follows:

```
python train_vq.py --id_num xxxx --train_set_file xxxx --test_set_file xxxx --gallery_set_file xxxx  --query_set_file xxxx --des_path xxxx --path_rst xxxx
```

* batch_size: the size of batch to be used for local training. (default: ```1024```)
* epoch_num: the number of total training epoches. (default: ```3000```)
* temp: the value of the tempture in our contrastive loss. (default: ```0.07```)
* weight1: the weight of cross-entropy loss. (default: ```0.8```)
* weight2: the weight of contrastive loss. (default: ```0.2```)
* com_weight: the weight of the traditional competition mechanism. (default: ```0.8```)
* id_num: the number of ids in the dataset.
* gpu_id: the id of training gpu.
* lr: the inital learning rate. (default: ```0.001```)
* redstep: the step size of learning scheduler. (default: ```500```)
* test_interval: the interval of testing.
* save_interval: the interval of saving.
* train_set_file: the path of training text file.
* test_set_file: the path of testing text file.
* gallery_set_file: the path of gallery text file.
* query_set_file: the path of query text file.
* des_path: the path of saving checkpoints.
* path_rst: the path of saving results.

Model-specific parameters can be found in the train file of each respective model.

Train files suffixed with 'vq' belong to the PalmBridge framework, while those without it correspond to the naive baseline.

#### Acknowledgments

Thanks to my all cooperators, they contributed so much to this work.
