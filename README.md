# RMT

An unofficial implementation of ["RMT: Retentive Networks Meet Vision Transformers](http://arxiv.org/abs/2309.11523). I created this repo to exercise my paper-to-code translation skill while waiting for the official implementation to be published on: https://github.com/qhfan/RMT.

## Introduction
![RMT](https://github.com/farrosalferro/RMT-unofficial/assets/127369839/d34050d9-c168-4dd8-ae3f-c07f611524d3)
**RMT** is an architecture that adopts the retention mechanism proposed by Sun et al. in the paper ["Retentive Network: A Successor to Transformer for Large Language Models"](http://arxiv.org/abs/2307.08621), which capably serves as a general-purpose backbone for computer vision. It extends the usability of retention mechanism from unidirectional, one-dimensional data (sequential data like texts) to bidirectional, two-dimensional data (images). Moreover, unlike the original Retentive Network, RMT does not apply the different-representation scenario for training and inference as the recurrent form greatly disrupts the parallelism of the model that results in a very slow inference speed.

RMT achieves strong performance on COCO object detection (`51.6 box AP` and `45.9 mask AP`) and ADE20K semantic segmentation (`52.0 mIoU`), surpassing previous models by a huge margin.

## Updates

### 28/11/2023
This repo was created by forking the mmpretrain repo ([mmpretrain](https://github.com/open-mmlab/mmpretrain)). Update the description inside the readme file.

## Installation

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
```

Please refer to [installation documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for more detailed installation and dataset preparation.

## Usage

to train the RMT model, you can use tools/train.py. Here is the full usage of the script:

```shell
python tools/train.py ${CONFIG_FILE} [ARGS]
```
where CONFIG_FILE is the path to the config file. There are some predefined config files available inside the `configs/rmt` folder. One example is `rmt-tiny_b128_cifar10.py` where it runs the tiny configuration of rmt with batch size of 128 of the CIFAR10 dataset. Please refer to these tutorials about the basic usage of MMPretrain for new users:

- [Learn about Configs](https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html)
- [Prepare Dataset](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html)
- [Inference with existing models](https://mmpretrain.readthedocs.io/en/latest/user_guides/inference.html)
- [Train](https://mmpretrain.readthedocs.io/en/latest/user_guides/train.html)
- [Test](https://mmpretrain.readthedocs.io/en/latest/user_guides/test.html)
- [Downstream tasks](https://mmpretrain.readthedocs.io/en/latest/user_guides/downstream.html)
- [MMPretrain Documentation](https://mmpretrain.readthedocs.io/en/latest/).

## Acknowledgement

MMPreTrain is an open source project that is contributed by researchers and engineers from various colleges and companies. Appreciation to all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. I also would like to thank the authors for writing such a wonderful paper.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{rmt-unofficial,
    title={RMT Unofficial Implementation},
    author={Farros Alferro},
    howpublished = {\url{https://github.com/farrosalferro/RMT-unofficial}},
    year={2023}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
