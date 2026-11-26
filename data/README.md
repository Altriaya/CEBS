# Input data

Acquire PEMS03 from the data supplied with [BackTime](https://github.com/xiaolin-cs/BackTime)
or [TimeGuard](https://github.com/qducnguyen/TimeGuard), ETTm1 from
[ETDataset](https://github.com/zhouhaoyi/ETDataset), and Weather through the data
references in [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
Follow their data terms. Attack generation is separate from the CEBS defense.

## Required NPZ inputs

| File | Keys | Convention |
| --- | --- | --- |
| Observed training | `series: (T,C)`, optional `marks: (T,4)` | possibly poisoned records, original units |
| Trusted pool | `x: (M,12,C)`, `y: (M,12,C)`, `starts: (M,)`, `block_ids: (M,)`, scalar `mean`,`std`, optional `marks: (M,12,4)` | x normalized, y original units |
| Clean validation | `series: (V,C)`, optional `marks: (V,4)` | original units, V >=24 |

Trusted starts are global physical indices in the validation interval; block IDs
are 0..11, with consecutive window starts inside each block and disjoint physical
blocks. The dedicated pool covers at most 5% of the training length in unique
physical time points. Shared clean validation and preprocessing statistics are
additional protocol resources, not disjoint percentages that can simply be added.
The caller must establish that the acquired pool is genuinely clean.

The infected-proxy checkpoint is the native FEDformer dictionary with `model`
(state dict) and `model_config`, matched to the observed series' channels and
normalization. Supply a distinct `channel_proxy` only when required by the recorded
protocol. Plain upstream checkpoints need a verified format conversion.

All three victims use the same scalar mean/std supplied by the trusted pool.
The core detector accepts no Oracle Binding/Input-only/Normal roles and does not
read test performance to select its thresholds.

## Optional evaluation

An evaluation NPZ contains `clean_x`, `trigger_x`, `truth`, `attack_target`,
`channels`, scalar `pattern_len`, and optional `x_marks`. Histories, predictions,
truth, and targets use original units. The same trigger/test windows and target
region must be used for the models being compared. Evaluation is run only after
the clean-validation-selected checkpoint exists.

Raw data, poisoned arrays, trained proxy/attacker weights, generated masks, and
training outputs are ignored by Git. The repository does not bundle the private
historical experiment registry or require its machine-local lineage files.
