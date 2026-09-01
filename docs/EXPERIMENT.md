# The experiment this trainer runs

**Status: not yet run.** The code is here and proven identical to its source. No
result exists.

## What it changes

One thing: the pixels the model trains on are distorted.

That sounds too small to be an experiment, so here is why it is one. The
pipeline this came from carried nine augmentation settings in its config, a flag
that set them, and a trainer that printed `augment=True` and wrote
`augmentation_enabled: true` into every checkpoint. **The dataset never read
those fields.**

Two independent causes, either one sufficient:

```text
the base class hard-coded train=False   not read from the config
the dataset wrote its own loader        going straight from DICOM to triplets,
                                        so the augmentation function was never
                                        on that code path at all
```

Measured rather than inferred. Building the dataset twice from the same DICOM
series, once with every augmentation field set and once with none:

```text
two draws with augmentation ON, identical to each other : True
augmentation ON identical to augmentation OFF           : True
maximum absolute difference                             : 0.0
```

Byte for byte identical, for 27 hours. That fact is pinned as
`test_the_plain_dataset_still_ignores_config_augmentation_fields`, which will
start failing if the inherited dataset ever begins augmenting on its own — at
which point augmentation would be applied twice.

## What it holds fixed

Everything else. The geometry, the head, the labels, the loss, the split, the
schedule, the learning rates and the seed are all the previous run's, so a
difference in the result is attributable to augmentation and nothing else.

```text
rotation        +/- 5 degrees        b7_rotation_deg
translate       +/- 3% per axis      b7_translate_frac
scale           +/- 5%               b7_scale_jitter
gamma           1 +/- 0.12           b7_gamma_jitter
bias field      +/- 0.08, clamped    b7_bias_field_strength
noise           std 0.02             b7_noise_std
slice dropout   8% of slices         b7_slice_dropout
```

Every value is read from `config/training.yaml`, not written into the code, so
none of them is a second change smuggled in.

Three small deliberate differences from the original augmentation function, each
because it now runs somewhere the original never did:

* **Each axis shifts by its own side.** Series are rectangles of roughly constant
  area, not squares. Scaling both shifts by one `image_size` would move a tall
  series much further sideways than up.
* **Slice dropout never empties a series.** At `p=0.08` over 32 slices this
  essentially never fires; the case it prevents is a blank study still carrying a
  real label, which teaches something false rather than nothing.
* **Draws come from an explicit generator**, seeded by run seed, epoch and study
  index, not the global random state. A run is reproducible, and DataLoader
  workers cannot repeat one another's numbers.

## What it deliberately leaves alone

Two of the nine configured settings change *which slices are chosen* rather than
what the chosen pixels look like: `center_jitter` and `train_gap_choices`. Slice
choice is a frozen contract — the centre function hard-codes `jitter=0` — and
moving it would be a second change in the same run.

`--slice-jitter N` implements it, is tested, and defaults to `0`.

## The check that gates a run

`preflight` draws the same study twice and **refuses to start** unless the two
tensors differ:

```text
[preflight] augmentation reaches the pixels: 2/2 series changed, max |diff| 0.318
[preflight] PASS encoder tensors with gradient=142
```

The measurement goes into every checkpoint as `augmentation_verified`, beside
the policy itself. A boolean nobody measured is what allowed the original
failure; recording the measurement instead is the fix.

`preflight` also refuses to start if any study's DICOM folders are missing, which
otherwise surfaces as `B42 study has no readable MRI series` from inside the
first epoch, naming neither the study nor the path.

## What it is measured against

The same run without augmentation, on the same 548 unseen-scanner studies:

```text
frozen baseline                       0.763117
trained encoder + finishing cosine    0.802666   1,447 training studies
trained encoder + finishing cosine    0.834998   3,801 training studies
```

The 3,801-study baseline is complete: six epochs, 26.9 hours, best at epoch 5.

```text
epoch   train      validation   macro AUC
  1     1.124810   1.075395     0.777063
  2     1.054743   1.028828     0.815093
  3     1.002861   1.049431     0.832568
  4     0.965207   1.013574     0.828500
  5     0.928085   1.010989     0.834998   <- selected
  6     0.903380   1.015765     0.833541
```

**That baseline is the reason to run this experiment.** Its last four epochs
span `0.0065`, so it has flattened rather than turned over. Train loss kept
falling the whole way, `-0.221` end to end, while validation loss went flat
after epoch 2 and validation AUC went flat with it. A model that keeps fitting
the training data while the held-out score stops moving is memorising, and more
epochs at these settings will not help.

Memorisation on a few thousand studies is what augmentation is for -- and that
run had none, whatever its own log said.

Those are **selection statistics** — each is the best of several epochs on the
very surface used to choose the epoch, so each is optimistically biased by
construction. They are comparable with each other and with this run, and with
nothing else. Not with a leaderboard score.

`--no-augment` reproduces the middle rows exactly, which is the control arm.

## Reading the result, decided in advance

* **Clearly above the baseline.** Augmentation was worth having, and the next
  questions are a longer schedule and stronger settings — a model that can no
  longer memorise usually tolerates both.
* **Level with the baseline.** Six epochs on 3,801 studies was not long enough
  for memorisation to be the binding constraint. Augmentation would then be
  expected to pay off only alongside a longer schedule.
* **Below the baseline.** The settings are too strong for this data, most likely
  the geometric ones on a task where small structures matter. Worth re-running
  with rotation and scale halved before concluding anything.

## What you need to run it

Four artefacts, none of them in this repository, because they are large and
unchanged:

```text
--data-root         the competition data folder, with train.csv,
                    train_series.csv and the DICOM directories
--labels-root       the merged report-label export: training_targets.csv,
                    policy.json, audit.json
--series-policy     series_policy.json, whose fingerprint the run verifies
--base-checkpoint   the pretrained base model this run starts from
--domain-split      the scanner-grouped split directory
```

The run verifies each of them before spending a GPU: the series policy by
fingerprint, the base checkpoint by its recorded arm and endpoint, the split by
the SHA-256 of the `train.csv` it was built from, and the data by counting how
many studies actually have readable series.

## Governance

Like the runs it is measured against, this selects its checkpoint on a held-out
split. That is competition practice and deliberately not a frozen-endpoint
policy. Its validation number is a selection statistic, not evidence of an
effect, and the checkpoint says so in its own `governance` field.
