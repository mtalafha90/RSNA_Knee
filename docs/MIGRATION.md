# How this package was extracted, and why that way

## The problem

The archive this came from held 270 modules named after the experiment that
first needed them. Running one training script imported 53 of them, 20,738
lines. Of the 482 definitions in that closure, **240 were ever called**. The
other half was ancestry: each model class inherited from the previous
experiment's class, so a run carried four generations of superseded code.

Reading it required knowing the history. `b42_constant_area_aspect_sparse_mil`
tells you which experiment introduced constant-area resizing; it does not tell
you that this is where the dataset lives.

## The method, and why not a rewrite

Retyping 5,000 lines of numerical code by hand would be the obvious approach and
the wrong one. The value of this code is that its measurements can be trusted,
and a rewrite that changes a number in one place makes every previous
measurement silently incomparable. There is no test that would catch it, because
the number would still look plausible.

So the migration is **mechanical**. `docs/migration_tool.py` is the script that
performed it, kept here as the record:

1. Walk the symbol graph from the training entry point. Keep only definitions
   that are actually reachable.
2. Assign each surviving definition a destination module by purpose.
3. Rename experiment-numbered names to descriptive ones.
4. Emit each module: source text sliced from the original, **comments and all**,
   in dependency order.

Step 4 slices text rather than regenerating from a syntax tree, because a tree
carries no comments and the comments are most of what makes the inherited code
readable.

## The rule that constrained everything

**Class and module names are free. `nn.Module` attributes are not.**

Checkpoints are loaded with `load_state_dict`, whose keys are attribute paths.
`self.encoder` and `self.global_projection` had to survive untouched or the
Phase-9 base checkpoint would stop loading. Class names never appear in a
state dict, so those were free.

## The guard that fired

A rename is a text substitution, so a name that also appears inside a string
literal gets rewritten there too. Sometimes that is right — an `__all__` entry
should follow the rename. Sometimes it would be a disaster: a frozen version
string or a fingerprint written into checkpoints and compared by contract checks
must never change.

The tool classifies rather than refusing blindly. It refuses only when the string
is assigned to a name ending in `_VERSION`, `_SIGNATURE`, `_SHA256`, `_DIGEST` or
`_POLICY`, or when it looks like a hex fingerprint. Everything else it renames
and **prints for review**.

It fired once, on `B50_ALWAYS_FROZEN_PREFIXES` inside an `__all__` list — a false
positive, and the reason the guard now distinguishes the two cases.

## The proof

`tests/test_migration_equivalence.py` runs both packages side by side on
identical inputs and compares bit for bit:

```text
the pixels    intensity normalisation, slice centres, crop/resize/triplets,
              a whole decoded study, the series index
the augment.  policy values, distorted pixels at three seeds, a whole
              augmented study
the model     base weights, state_dict keys, logits, local logits,
              optimiser groups
the maths     masked targets, macro AUC, fast_auc, split selection
```

**25 of 25 passed.** Set `RSNA_KNEE_ORIGINAL` to the original `developments/src`
and run pytest to reproduce it. Without that variable the suite skips, which is
the normal case on a machine that only has this repository.

## What was deliberately left behind

```text
270 -> 26 modules          only what the training path reaches
20,738 -> 6,011 lines      the ancestry nothing calls is gone
482 -> 240 definitions     half the closure was dead
```

Also left behind: every other experiment's trainer, submission launcher,
evaluation script and audit tool. They remain in the original archive. This
repository is one code path, said plainly.
