<div align="center">

# Pantheon Lab Programming Assignment

</div>

---

## Part 1 — GAN MNIST: Observations & Implementation Notes

### Environment Setup

**Difficulty 1 — `jupyterlab` install failure**

Running `pip install -r requirements.txt` on the provided `pantheon-py38` (Python 3.8) conda environment failed because `jupyterlab`'s build chain pulls in `puccinialin`, which requires Python ≥ 3.9:

```
ERROR: Could not find a version that satisfies the requirement puccinialin (from versions: none)
ERROR: No matching distribution found for puccinialin
```

`jupyterlab` is not used anywhere in the codebase (no `.ipynb` notebooks, no code imports), so it was commented out in `requirements.txt`.

**Difficulty 2 — SSL error when downloading MNIST**

After fixing the install, `trainer.fit()` crashed during `MNISTDataModule.prepare_data()` with:

```
ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data (_ssl.c:4194)
```

Python's `ssl` module on this Windows + conda 3.8 setup crashes while loading the system CA store, even when `SSL_CERT_FILE` is pointed at the certifi bundle. The fix was to force-replace the default HTTPS context in `mnist_datamodule.py` before any download occurs:

```python
import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
```

`certifi` and `requests` were added to `requirements.txt`.

---

### Implementation

| Area | What was done |
|------|--------------|
| **Hydra config** | Filled in `configs/model/mnist_gan_model.yaml` — instantiated `Generator` and `Discriminator` via `_target_` with OmegaConf relative interpolation (`${..n_classes}` etc.) so all shared params stay single-sourced |
| **GAN training step** | Implemented alternating generator / discriminator optimisation in `step()` using `optimizer_idx`; discriminator uses `.detach()` on generator output to prevent gradient leakage |
| **W&B logging** | Loss scalars (`g_loss`, `d_loss`, `d_real_loss`, `d_fake_loss`) logged each step via `self.log_dict`; `on_epoch_end` logs a `torchvision.make_grid` image (2 rows × 5 cols, one digit class per cell) to W&B at `step=self.current_epoch` |
| **Test evaluation** | Implemented `test_step` and made `trainer.test()` explicit in `train.py`, loading from the best checkpoint when available |
| **Training length** | Set `max_epochs: 20` and `max_steps: -1` in the experiment config; aligned `ModelCheckpoint.every_n_train_steps` to ~1 checkpoint per epoch (1720 steps) instead of the original 10 000 |

---

### AI Suggestions Rejected / Corrected

1. **Adding `certifi` + `requests` to requirements alone** — AI suggested this would fix the SSL error, but installing the packages without patching the default context had no effect (`urllib` still loaded the broken Windows cert store). Rejected; the actual fix required monkey-patching `ssl._create_default_https_context` in the datamodule.

2. **Bumping `max_epochs` from 10 → 20 without `max_steps: -1`** — AI made the epoch change but left an implicit step ceiling in place. Rejected as incomplete; required explicitly setting `max_steps: -1` to ensure epoch count alone governs training length.

---

### Results — 20-Epoch Run

**Final metrics (W&B run summary)**

| Metric | Train | Test |
|--------|-------|------|
| `g_loss` | 0.367 | 0.393 |
| `d_loss` | 0.194 | 0.228 |
| `d_real_loss` | 0.198 | 0.303 |
| `d_fake_loss` | 0.191 | 0.152 |
| `global_step` | 17 200 | — |

**Loss curves**

![loss_curves](./images/loss_curves.png)

Generator loss (`train/g_loss`) and discriminator loss (`train/d_loss`) both decrease and stabilise after ~5 k steps, indicating the training converged. The near-equal `d_real_loss` and `d_fake_loss` at the end of training shows the discriminator is roughly balanced between real and generated samples.

**Generated samples progression**

| Early training | Mid training | Epoch 20 (final) |
|:-:|:-:|:-:|
| ![gen_early](./images/gen_early.png) | ![gen_mid](./images/gen_mid.png) | ![gen_epoch20](./images/gen_epoch20.png) |

Each grid shows one generated sample per digit class (0–9), arranged in two rows. Digit structure becomes clearly recognisable by epoch 20, consistent with the example in the assignment brief.

---

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

AI assistance is allowed and expected. Submit the complete raw session export or native
conversation log produced by your AI tool, and identify at least 3 moments where you
rejected, corrected, or substantially reworked an AI suggestion.

## What is all this?
This "programming assignment" is really just a way to get you used to
some of the tools we use every day at Pantheon to help with our research.

There are 4 fundamental areas that this small task will have you cover:

1. Getting familiar with training models using [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)

2. Using the [Hydra](https://hydra.cc/) framework

3. Logging and reporting your experiments on [weights and biases](https://wandb.ai/site)

4. Showing some basic machine learning knowledge

## What's the task?
The actual machine learning task you'll be doing is fairly simple! 
You will be using a very simple GAN to generate fake
[MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) images.

We don't excpect you to have access to any GPU's. As mentioned earlier this is just a task
to get you familiar with the tools listed above, but don't hesitate to improve the model
as much as you can!

## What you need to do

To understand how this framework works have a look at `src/train.py`. 
Hydra first tries to initialise various pytorch lightning components: 
the trainer, model, datamodule, callbacks and the logger.

To make the model train you will need to do a few things:

- [ ] Complete the model yaml config (`model/mnist_gan_model.yaml`)
- [ ] Complete the implementation of the model's `step` method
- [ ] Implement logging functionality to view loss curves 
and predicted samples during training, using the pytorch lightning
callback method `on_epoch_end` (use [wandb](https://wandb.ai/site)!) 
- [ ] Answer some questions about the code (see the bottom of this README)

**All implementation tasks in the code are marked with** `TODO`

Don't feel limited to these tasks above! Feel free to improve on various parts of the model

For example, training the model for around 20 epochs will give you results like this:

![example_train](./images/example_train.png)

## Getting started
After cloning this repo, install dependencies
```yaml
# [OPTIONAL] create conda environment
conda create --name pantheon-py38 python=3.8
conda activate pantheon-py38

# install requirements
pip install -r requirements.txt
```

Train model with experiment configuration
```yaml
# default
python run.py experiment=train_mnist_gan.yaml

# train on CPU
python run.py experiment=train_mnist_gan.yaml trainer.gpus=0

# train on GPU
python run.py experiment=train_mnist_gan.yaml trainer.gpus=1
```

You can override any parameter from command line like this
```yaml
python run.py experiment=train_mnist_gan.yaml trainer.max_epochs=20 datamodule.batch_size=32
```

The current state of the code will fail at
`src/models/mnist_gan_model.py, line 29, in configure_optimizers`
This is because the generator and discriminator are currently assigned `null`
in `model/mnist_gan_model.yaml`. This is your first task in the "What you need to do" 
section.

## Open-Ended tasks (Bonus for junior candidates, expected for senior candidates)

Staying within the given Hydra - Pytorch-lightning - Wandb framework, show off your skills and creativity by extending the existing model, or even setting up a new one with completely different training goals/strategy. Here are a few potential ideas:

- **Implement your own networks**: you are free to choose what you deem most appropriate, but we recommend using CNN and their variants if you are keeping the image-based GANs as the model to train
- **Use a more complex dataset**: ideally introducing color, and higher resolution
- **Introduce new losses, or different training regimens**
- **Add more plugins/dependecy**: on top of the provided framework
- **Train a completely different model**: this may be especially relevant to you if your existing expertise is not centered in image-based GANs. You may want to re-create a toy sample related to your past research. Do remember to still use the provided framework.

## Questions

Try to prepare some short answers to the following questions below for discussion in the interview.

* What is the role of the discriminator in a GAN model? Use this project's discriminator as an example.

* The generator network in this code base takes two arguments: `noise` and `labels`.
What are these inputs and how could they be used at inference time to generate an image of the number 5?

* What steps are needed to deploy a model into production?

* If you wanted to train with multiple GPUs, 
what can you do in pytorch lightning to make sure data is allocated to the correct GPU? 

## Submission

- Using git, keep the existing git history and add your code contribution on top of it. Follow git best practices as you see fit. We appreciate readability in the commits
- Add a section at the top of this README, containing your answers to the questions, as well as the output `wandb` graphs and images resulting from your training run. You are also invited to talk about difficulties you encountered and how you overcame them
- Link to your git repository in your email reply and share it with us/make it public

## Chatbot Assignment

Compare at least **3 LLMs**, each running locally on your device through
[Ollama](https://ollama.com/) or another local model server suitable for your hardware. A GPU is not required.

For the model-comparison evidence, record each model, complete prompt (including relevant
system instructions), parameter settings, and unedited response. Clearly label each test
so it can be matched to your analysis, and do not include API keys or other secrets.

- Compare the models across content quality, contextual understanding, language fluency,
  and ethical considerations. Support your conclusions with examples from your tests.
- Explain the parameters used to control model responses and how each affects behavior.
- Explore prompt-engineering techniques such as template-based, rule-based, and
  machine-learning-based prompting. Discuss their challenges and trade-offs with examples.
- Explain retrieval-augmented generation (RAG) and how it is applied to natural-language
  generation tasks.

<br>
