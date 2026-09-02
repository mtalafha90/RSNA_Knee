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

## The second reading, which the last hidden result made necessary

The 548-study validation surface this run selects its epoch on is
**report-derived**, like the training labels. The hidden competition test is
scored against **expert** labels. Those are not the same target, and the gap
has now been measured three times:

```text
B15    +0.167 teacher agreement    -0.008 expert AUC
B25X   +0.058 weak surface         +0.002 on eleven targets
B50    +0.011 report-derived       -0.002 expert-58   ->  -0.001 hidden
```

That last row is the reason this section exists. B50's report-derived
comparison was well run and passed a rule frozen in advance. Its expert-58
audit disagreed and was recorded as inconclusive, correctly, because 58
studies resolve to about `+/-0.03`. Carried to full scale as B51, the hidden
test returned `-0.001` -- closer to the audit than to the 548-study surface.

So this run is read twice:

```bash
rsna-knee-expert-audit --checkpoint runs/augmented_training/best_model.pt \
                       --label b53

rsna-knee-expert-audit --checkpoint <the no-augment control> --label control

rsna-knee-expert-audit --checkpoint runs/augmented_training/best_model.pt \
                       --label b53 \
                       --against artefacts/expert58/control_expert58_predictions.csv
```

The audit reports the delta **and its ceiling** -- the fraction of
positive-negative study pairs the two checkpoints order differently, which is
exactly the largest AUC difference they could possibly show. B48 and B49 were
both judged against a `+0.010` threshold with ceilings of `0.0015` and
`0.0024`: neither could have passed whatever its mechanism did. A delta
without its ceiling hides that.

**Read the audit as a veto, never as a confirmation.** Fifty-eight studies
cannot support a small gain; they can only catch a report-surface gain that
fails to reach expert truth, which is what has happened three times. The rule
is fixed in `expert_audit.py` before any B53 number exists:

```text
delta <= -0.020    VETO. Do not submit; this is B15's failure again.
-0.020 to +0.010   INCONCLUSIVE, the likely and acceptable outcome. The
                   result then rests on the report-derived comparison alone,
                   and the record must say so.
delta >= +0.010    supported on expert truth as well.
```

The 58 studies are held out of every run and stay that way. This audit selects
nothing -- not an epoch, not a seed, not a setting. Selecting on them would
spend the only expert-truth proxy the project has.

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

## Feeding the GPU

`num_workers: 0` in `config/training.yaml` decodes DICOM and applies all seven
augmentations on the main thread, between GPU steps, so a fast card waits for
the CPU. Augmentation makes that worse than it was without it.

Raising it is safe:

```yaml
num_workers: 6
prefetch_factor: 2
```

**It changes throughput and nothing else**, which matters because the baseline
above was measured at `0`. Three design decisions make that true. Two of them
`tests/test_loader_workers.py` checks by running a real loader at 0 and at 2
workers and comparing the studies and their pixels:

* the shuffle order comes from an explicitly seeded generator, not the global
  random state, so every worker count visits studies in the same order;
* the augmentation draws from a generator keyed by run seed, epoch and study
  index, so workers cannot inherit copies of one stream and hand out the same
  "random" numbers.

The third is how tensors cross between processes, and it is there because the
first run at `num_workers: 6` trained a whole epoch and then died at the first
validation pass:

```text
File ".../torch/multiprocessing/reductions.py", rebuild_storage_fd
RuntimeError: received 0 items of ancdata
```

That is the process running out of file descriptors, not a fault in the data.
A study item here is a *list* of per-series tensors, so one batch is a dozen or
more separate shared allocations, and under the Linux default every one of them
travels as an open descriptor. Enough batches in flight across enough workers
and the limit is reached -- which is why it struck at validation rather than at
startup. Each worker now shares by named shared-memory file instead, and the
run raises its own soft descriptor limit as well. The log states both:

```text
workers=6 | sharing=file_system | open_files=1048576
```

`tests/test_worker_file_descriptors.py` runs a real loader under a deliberately
small limit, in two arms: the package's worker setup must survive it, and the
setup from before the fix must not. The second arm is what stops the first from
quietly becoming a test that 200 items load.

The one cost of the change: a hard kill can leave stray files in `/dev/shm`.

`num_workers: 0` is left exactly as it was -- it shares nothing between
processes, so neither setting applies, and the run that produced the baseline
above stays reproducible.

Each worker is a separate process under `spawn` and costs host RAM. Preflight
after changing it and read the reported `rss`.

Whether the extra workers earn their keep is a separate question from whether
they are safe. If `nvidia-smi` already shows the card near 100% busy, the CPU is
not the constraint and more workers will buy little.

## Governance

Like the runs it is measured against, this selects its checkpoint on a held-out
split. That is competition practice and deliberately not a frozen-endpoint
policy. Its validation number is a selection statistic, not evidence of an
effect, and the checkpoint says so in its own `governance` field.
