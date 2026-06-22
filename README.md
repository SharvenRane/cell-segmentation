# cell-segmentation

A small baseline for cell instance segmentation. A compact UNet predicts a
binary foreground mask, and a watershed step turns that mask into separate
instances so that touching cells are counted as two rather than one. A synthetic
generator keeps the tests fast and download free, and `src/dsb_data.py` loads the
real 2018 Data Science Bowl nuclei set for an actual training run (see Results).

## Why two stages

A binary mask answers "where are the cells" but not "how many". When two
cells touch, their pixels merge into one connected blob and a naive count
sees a single object. The watershed stage fixes that. It computes the
distance transform of the mask, which peaks near each cell center, seeds a
marker at every peak, and floods the foreground from those seeds. The flood
fronts meet along the thin neck between cells, which is where the split
lands.

## Layout

```
src/
  model.py        compact UNet (encoder, decoder, skip connections)
  data.py         synthetic blob generator and a two touching circles case
  postprocess.py  watershed split into instances plus an instance counter
  metrics.py      Dice score, soft Dice loss, instance count error
  train.py        a tiny overfit loop on synthetic blobs
tests/            pytest behavior tests
```

## The pieces

**Model.** `UNet` is a two level encoder decoder with skip connections and a
1x1 head that emits a single channel logit map. The base feature width is a
constructor argument so it stays small for tests.

**Data.** `make_blob_sample` paints bright circular cells on a dark
background and returns the image, a binary mask, and an integer instance
map. `make_two_touching_circles` builds the deliberately hard case where two
disks overlap into one connected component.

**Post processing.** `watershed_instances` runs the distance transform,
finds peaks, and floods with watershed. `count_instances` counts distinct
positive ids.

**Metrics.** `dice_coefficient` scores binary overlap, `soft_dice_loss` is
the differentiable training objective, and `instance_count_error` reports
the absolute gap between predicted and true cell counts.

## Running

Install the requirements into your environment and run the tests:

```
pip install -r requirements.txt
pytest tests/ -q
```

A short training demo lives in `src/train.py`. `train_overfit` fits the UNet
to a fixed batch of easy synthetic masks and returns the initial loss, the
final loss, and the trained model.

## Results on real nuclei (DSB2018)

`src/dsb_data.py` reads the 2018 Data Science Bowl nuclei set (BBBC038) from a
free community mirror, no credentials needed. A 40 epoch UNet run on 531 training
and 133 validation images at 256 px reached a best validation Dice of 0.901 on an
RTX 5070 Ti. Real microscopy with real masks; the synthetic generator stays in
place so the test suite needs nothing downloaded.

## What the tests check

These are behavior checks, not fixed magic numbers.

* Dice is 1.0 for identical masks, near 0 for disjoint masks, and matches
  the closed form value for a known half overlap.
* The soft Dice loss is low when the prediction is confident and correct,
  and its gradient is finite.
* Watershed splits the single connected blob from two touching circles into
  exactly two instances, leaves a single circle as one instance, and returns
  zero instances for an empty mask. No label leaks onto background pixels.
* The UNet preserves spatial shape from input to output.
* The overfit loop drives the loss down, and the resulting model reaches a
  Dice above 0.9 on an easy synthetic mask.
* The full predict then split pipeline recovers a cell count close to the
  truth.

All numbers asserted in the tests come from runs on the synthetic data in
this repo. Nothing here is a quoted benchmark.

## Notes

The synthetic generator is intentionally simple so the tests are fast and
deterministic with seeded random generators. `src/dsb_data.py` is the real
counterpart: it yields the same image, mask, and instance triples from the
DSB2018 nuclei set, so the model, metrics, and watershed stages stay exactly as
they are whether the data is synthetic or real.
