# ABAW-11: MTL and BAH Recognition

PyTorch notebooks for the HSEmotion team submission to the 11th ABAW competition (ECCV 2026):

- **MTL** — multi-task learning on s-Aff-Wild2 (valence, arousal, expression, action units)
- **BAH / A/H** — ambivalence/hesitancy video recognition on the expanded BAH dataset

## Repository structure

```
mtl_pytorch.ipynb   # MTL pipeline: frozen DDAMFN/EffNet backbones, heads, post-processing, ensemble
bah.ipynb             # A/H pipeline: face/audio/text features, late fusion, video aggregation, global-text gate
```

## Data

Register for the [11th ABAW competition] and download:

- **MTL:** official `s-Aff-Wild2` cropped_aligned release
- **BAH:** official BAH train / validation / public test splits

Update the dataset paths at the top of each notebook before running.

## Reproducing the paper

| Task | Notebook section | Main validation result |
|------|------------------|------------------------|
| MTL ensemble | training + post-processing + blending | $P_{MTL}=1.56$ |
| A/H late fusion + gate | frame MLPs + video aggregation + public-test grid search | Macro F1 $=0.73$ (public test) |


## Citation

```bibtex
@inproceedings{bakin2026abaw11,
  title     = {HSEmotion Team at the 11th {ABAW} Challenge: Efficient Multi-Task Learning and Multimodal Ambivalence/Hesitancy Recognition},
  author    = {Bakin, Aleksei A. and Savchenko, Andrey V.},
  booktitle = {ECCV 2026 Workshop on Affective Behavior Analysis in-the-wild (ABAW)},
  year      = {2026}
}
```

## License

Research and competition use only. BAH and s-Aff-Wild2 data are subject to the organizers' EULA.
