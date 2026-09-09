"""Code of my own that anemoi loads by name from a config `_target_`.

Named xbris, not bris, because `bris` is already taken: metno/bris-inference
installs a top-level package by that name into the same environment, and a
second one would shadow it depending on sys.path order.

Anything here must be importable from the training job, which means the repo
root has to be on PYTHONPATH. bris/slurm/finetune.sbatch puts it there.
"""
