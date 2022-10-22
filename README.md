# Secrets of Event-Based Optical Flow (ECCV 2022)

This is the official repository for [**Secrets of Event-Based Optical Flow**](https://arxiv.org/abs/2207.10022), **ECCV 2022 Oral** by  
[Shintaro Shiba](http://shibashintaro.com/), [Yoshimitsu Aoki](https://aoki-medialab.jp/aokiyoshimitsu-en/) and [Guillermo Callego](https://sites.google.com/view/guillermogallego).

 <!-- - [Paper]() -->
[[Video](https://youtu.be/nUb2ZRPdbWk)] [[PDF](https://arxiv.org/pdf/2207.10022)]

[![Secrets of Event-Based Optical Flow](docs/img/thumbnail_github.png)](https://youtu.be/nUb2ZRPdbWk)


If you use this work in your research, please cite it (see also [here](#citation)):

```bibtex
@InProceedings{Shiba22eccv,
  author        = {Shintaro Shiba and Yoshimitsu Aoki and Guillermo Gallego},
  title         = {Secrets of Event-based Optical Flow},
  booktitle     = "European Conference on Computer Vision (ECCV)",
  year          = 2022
}
```

### Optional tasks (for me)

**The code here is already runnable, and explains the ideas of the paper enough.** (Please report bugs if any.)

Rather than releasing all of my (sometimes too experimental) codes,
I published just a minimal set of the codebase to reproduce.
So the following tasks are more optional for me.
But if it helps you, I can publish other parts as well. For example:

 - Other data loader

 - Some other cost functions

 - Pretrained model checkpoint file

 - Other solver (especially DNN)

 - The implementation of [the Sensors paper]((https://www.mdpi.com/1424-8220/22/14/5190))

Your feedback is helpful to prioritize the tasks, so please contact me or raise issues.
The code is modularized well, so if you want to contribute, it should be easy too.

-------
# Setup

## Requirements

- python 3.9

Python 3.8 should be fine although it is not strictly tested.
If you successfully run it on 3.8, please let me know so that I modify this readme.

GPU is entirely optional.
If `torch.cuda.is_available()` then it automatically switches to use GPU.
I'd recomment to use GPU for time-aware solutions, but CPU is ok for no-timeaware method as long as I tested.

### Tested environments

- Mac OS Monterey (both M1 and non-M1)
- Ubuntu (CUDA 11.1 or 11.3)

## Installation

I strongly recommend to use venv: `python3 -m venv <new_venv_path>`
Also, you can use [poetry]().

- Install pytorch and torchvision for your environment. Make sure you install the correct CUDA version if you want to use it.

- If you use poetry, `poetry install`. If you use only venv, check dependecy libraries and install it from [here](./pyproject.toml).

## Download dataset

Download each dataset under `./datasets` directory.
Optionally you can specify other root directory:
please check the [dataset readme](./datasets/README.md) for the details.

# Execution

```shell
python3 main.py --config_file ./configs/mvsec_indoor_no_timeaware.yaml
```

If you use poetry, simply add `poetry run` at the beginning.
Please run with `-h` option to know more about the other options.

## Config file

The config (.yaml) file specifies various experimental settings.
Please check and change parameters as you like.
# Citation

If you use this work in your research, please cite it as follows:

```bibtex
@InProceedings{Shiba22eccv,
  author        = {Shintaro Shiba and Yoshimitsu Aoki and Guillermo Gallego},
  title         = {Secrets of Event-based Optical Flow},
  booktitle     = "European Conference on Computer Vision (ECCV)",
  year          = 2022
}
```

This code also includes some implementation of the [following paper about event collapse in details](https://www.mdpi.com/1424-8220/22/14/5190).
Please check it :)

```bibtex
@Article{Shiba22sensors,
  author        = {Shintaro Shiba and Yoshimitsu Aoki and Guillermo Gallego},
  title         = {Event Collapse in Contrast Maximization Frameworks},
  journal       = {Sensors},
  year          = 2022,
  volume        = 22,
  number        = 14,
  pages         = {1--20},
  article-number= 5190,
  doi           = {10.3390/s22145190},
  url={https://www.mdpi.com/1424-8220/22/14/5190}
}
```

# Author

Shintaro Shiba [@shiba24](https://github.com/shiba24)

## LICENSE

Please check [License](./LICENSE).

## Acknowledgement

I appreciate the following repositories for the inspiration:

- [autograd-minimize](https://github.com/brunorigal/autograd-minimize)

-------
# Additional Resources

* [Event Collapse in Contrast Maximization Frameworks](https://www.mdpi.com/1424-8220/22/14/5190) ([arxiv](https://arxiv.org/pdf/2207.04007))
* [Research page (TU Berlin RIP lab)](https://sites.google.com/view/guillermogallego/research/event-based-vision)
* [Course at TU Berlin](https://sites.google.com/view/guillermogallego/teaching/event-based-robot-vision)
* [Survey paper](http://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)
* [List of Resources](https://github.com/uzh-rpg/event-based_vision_resources)
