# More Powerful Selective Kernel Tests for Feature Selection

This repository contains resources for selecting statistically significant
features using multiscale bootstrap. The test corrests for selection bias.
The algorithm is described in our [paper](https://arxiv.org/abs/1910.06134),

    Lim, J., Yamada, M., Jitkrittum, W., Terada, Y., Matsui, S., Shimodaira, H.
    More Powerful Selective Kernel Tests for Feature Selection
    AISTATS 2020

## How to install?

Requires ```numpy```, ```matplotlib```, ```SciPy```, ```sklearn```. The package
can be installed with the following command

    pip install git+https://github.com/jenninglim/multiscale-features

Once installed, you should be able to do `import mskernel` without any error.

## Demo

See ```notebooks```.

## Reproducing results

See ```experiments``` for experiment setup and its corresponding 
figures can be seen in ```figures```.

## See also

* Kernel Multiple Model Comparison: [Code](https://github.com/jenninglim/model-comparison-test)
[Paper](https://arxiv.org/abs/1910.12252).
* Kernel Goodness of Fit (where some of the kernel code is from): [Code](https://github.com/wittawatj/kernel-gof), [Paper](https://arxiv.org/abs/1705.07673)


