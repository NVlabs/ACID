[![NVIDIA Source Code License](https://img.shields.io/badge/license-NSCL-blue.svg)](https://github.com/NVlabs/ACID/blob/master/LICENSE)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# ACID: Action-Conditional Implicit Visual Dynamics for Deformable Object Manipulation

### [Project Page](https://b0ku1.github.io/acid/) | [Paper](https://arxiv.org/abs/2203.06856)

<div style="text-align: center">
<img src="_media/model_figure.png" width="600"/>
</div>

This repository contains the codebase used in [**ACID: Action-Conditional Implicit Visual Dynamics for Deformable Object Manipulation**](https://b0ku1.github.io/acid/). Specifically, the repo contains code for:
* [**PlushSim**](./PlushSim/), the simulation environment used to generate all manipulation data.
* [**ACID model**](./ACID/), the implicit visual dynamics model's model and training code.

If you find our code or paper useful, please consider citing
```bibtex
@article{shen2022acid,
  title={ACID: Action-Conditional Implicit Visual Dynamics for Deformable Object Manipulation},
  author={Shen, Bokui and Jiang, Zhenyu and Choy, Christopher and J. Guibas, Leonidas and Savarese, Silvio and Anandkumar, Anima and Zhu, Yuke},
  journal={Robotics: Science and Systems (RSS)},
  year={2022}
}
```

# ACID model
Please see the [README](./ACID/README.md) for more detailed information.


# PlushSim
Please see the [README](./PlushSim/README.md) for more detailed information.


# License
Please check the [LICENSE](./LICENSE) file. ACID may be used non-commercially, meaning for research or evaluation purposes only. For business inquiries, please contact researchinquiries@nvidia.com.
