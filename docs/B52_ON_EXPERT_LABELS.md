# B52 on expert labels

**Date:** 2026-09-02
**Status:** COMPLETED MEASUREMENT. The baseline every later checkpoint is
compared against, and the first evidence that this surface is worth having.

## The number

```text
macro AUC on 58 expert studies, 336 series    0.678247
```

## Three rulers, one model

```text
report labels, 548 studies    0.8350     what the trainer selects on
expert labels, 58 studies     0.6782     this audit
hidden test, ~1300 studies    0.7160     the competition
```

**The 58-study expert surface lands within `0.038` of the hidden score. The
548-study report surface is `0.119` away.** On one comparison that is suggestive
rather than proven, but it is the comparison the whole audit was built to make,
and it points the way the argument predicted: the hidden test scores agreement
with an expert, and so does this.

That makes it usable for what it was built for. A teacher change moves the
report-derived surface, because the labels on it are what changed. It does not
move this one.

## Per target, worst first

```text
  MCL               0.5011
  ACL               0.5478
  Lateral OA        0.5725
  Contusion         0.5735
  Fracture          0.6139
  PF OA             0.6512
  Lateral Meniscus  0.6783
  Medial Meniscus   0.7188
  Synovitis         0.7778
  Medial OA         0.7907
  Baker's           0.8152
  Effusion          0.8981
```

**MCL is a coin flip. ACL is barely off one.** The two cruciate and collateral
ligament findings -- among the most clinically consequential things a knee MRI
is read for -- carry essentially no signal against expert truth.

At the other end sit Effusion and Baker's cyst, both fluid, both bright on T2,
both large. The ordering reads as a size-and-contrast ranking rather than a
clinical one: the model finds what is big and bright, and does not find what is
small and grey.

## How much of this is noise

A great deal, per target. With 58 studies and roughly 15 positives for a
finding, an AUC carries a standard error near `+/-0.13`, so any single row above
is compatible with a range of about `+/-0.25` at 95%. **No individual target
here is evidence on its own.**

The pattern is worth more than any row in it. Two ligaments both at chance and
two fluid collections both above `0.81` is a shape, and it agrees with what the
geometry would predict: a constant-area 448-pixel resize and a 6x6 evidence grid
give a torn ACL very few cells to be found in.

## What it changes

```text
macro now                            0.6782
macro if ACL and MCL reached 0.70    0.7075
```

Lifting two targets off the floor is worth `+0.029` of macro AUC -- larger than
the entire spread between the five architectures this project has submitted
(`0.694` to `0.716`).

That is not a promise that it can be done. It says where the room is, which is
not where the last several experiments looked for it.

## Governance

The 58 expert studies are held out of every run and were read here only. This
audit selected nothing: no epoch, no seed, no setting. It is a veto on a
report-derived gain, never a confirmation of one, and it must not become a
surface anything is tuned on -- that would spend the only expert-truth proxy the
project has.
