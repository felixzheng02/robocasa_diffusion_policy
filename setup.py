from setuptools import setup, find_namespace_packages

# find_packages() returns [] here and always has: there are 159 .py files but only 7
# __init__.py, none of them at depth 1 or 2, so the classic finder sees no packages at all
# and `pip install -e .` installs NOTHING while reporting success. That is why every script
# in this repo has historically resolved `diffusion_policy` from the current working
# directory instead of from an install.
#
# The include filter is not optional. Bare find_namespace_packages() returns 119 entries,
# 60 of which are not diffusion_policy at all -- `outputs`, `outputs.pick_skill.checkpoints`,
# `wandb.*`, `media`, `tests` -- so it would attempt to package the 79 GB of training
# checkpoints under outputs/.
setup(
    name="diffusion_policy",
    packages=find_namespace_packages(include=["diffusion_policy*"]),
)
