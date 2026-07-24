# Learning track

A syllabus and an examiner for this repo.

Three tracks run in parallel. [`docs/ROADMAP.md`](../docs/ROADMAP.md) owns the
research arc and the GitHub issue numbering. [`PLAN.md`](../PLAN.md) owns the
lifecycle and legibility arc. This one owns understanding, and its labs are drawn
from the open work in the other two, so studying it moves the project rather than
sitting beside it.

## Governing principle

**The repo is the textbook. This track is the syllabus and the exam.**

No module here restates what a docstring already explains. The prose in
`backtest/models.py`, `backtest/splits.py`, `backtest/metrics.py`,
`ml/cold_start.py`, and `recommendations/scorer.py` is already the teaching
material; a second copy would drift within a month. Each module is framing, a
reading list of real files, a question set, and a lab. The explanation lives in
the answers, where it is allowed to be substantive.

## Using it

```bash
wowfc learn status              # mastery per module, cards due, lab state
wowfc learn next                # drill what is due, then new material
wowfc learn next --list         # see the queue without answering anything
wowfc learn module m06          # objectives, reading list, lab
wowfc learn exam -m m06 -n 10   # scored, nothing revealed until the end
wowfc learn lab lab-01-purge-embargo
wowfc learn validate            # check every citation against current code
```

Progress lives in `data/learn/progress.db`, which is gitignored. Set
`WOWFC_LEARN_DB` to point somewhere else. Nothing here writes to the product
database: that file is copied into every durable backup and feeds the M3
warehouse, and review state belongs in neither.

Spaced repetition is an SM-2 variant with four grades. A passing grade schedules
the card at least a day out, so re-running a drill the same day serves no repeat
of anything graded `good` or better. A card graded `again` stays due today and
does come back, which is relearning working as intended. Re-drilling a module
repeatedly in one afternoon will not push cards months out: the last grade of the
day replaces the earlier ones rather than compounding on them.

## Question kinds

| kind | what it asks | how it is scored |
|---|---|---|
| `recall` | free response | self-graded against a shown rubric, so the score is countable rather than a vague sense of having known it |
| `mcq` | pick one | auto-scored; every wrong option carries a note on why it is wrong, which is where most of the learning in this format happens |
| `predict` | given this code or config, what happens | auto-scored when it has options |
| `find` | go into the repo and locate something | self-graded; the cure for not remembering what is in here |

## The drift guard

Every citation is a file path plus a verbatim single-line **anchor**, never a line
number. A line number is wrong the moment a line is inserted above it.

```toml
source = "wow_forecaster/ml/trainer.py"
anchor = "val_split_date = date_strs[-(val_days + 1)]"
```

`tests/test_learning/test_bank_integrity.py` and `wowfc learn validate` both call
`check_content()` in `wow_forecaster/learning/integrity.py`, so the rule the test
enforces and the rule the authoring command reports are the same rule. It checks
that every source path resolves, every anchor still appears in its file (read as
UTF-8, because the quoted docstrings contain real Unicode), every `see_also`
reference and heading fragment resolves, and every referenced lab has a brief.

The consequence: editing a cited line in `trainer.py` turns CI red until the
question is updated. That is the point. A study guide that quietly describes last
month's code is worse than none.

Commit SHA references are checked only when the clone is deep enough to resolve
them. CI uses `actions/checkout@v4` with no `fetch-depth`, so it gets a depth-1
clone where no historical SHA resolves, and asserting there would fail every run
for a reason unrelated to accuracy.

## Layout

```
curriculum.toml          all 20 modules: title, objectives, reading list, prereqs, lab
modules/m06-*.md         one page of framing per module
banks/m06.toml           question banks
labs/lab-01-*.md         lab briefs
```

Banks land one part per PR. The curriculum declares all twenty modules from the
start so the shape of the track is visible; a module without a bank reports as
"not authored yet" rather than being hidden. A bank with no declared module is an
integrity failure, since nothing would ever serve it.

## The four parts

**Part I. The domain and the data** (M01 to M03). What is being forecast, how an
API response becomes a row, and how irregular observations become a daily series.

**Part II. Features, models, statistics** (M04 to M10). The leakage boundary,
baselines and metrics, splitting time series correctly, the model itself,
uncertainty, cold-start transfer, and turning a forecast into a decision.

**Part III. Failure, history, operations** (M11 to M14). The 105-day silent
outage, every bug grouped by failure class, seventy versions as a design record,
and capture that does not need the desktop.

**Part IV. Proving it and shipping it** (M15 to M20). The realization ledger,
significance testing, paper-trading P&L, causal designs for the event study, the
lifecycle gap, and both roadmaps.

## The labs

Real branches, real PRs, house rules. Each brief names the branch, the issue, the
acceptance test to write first, the files to touch, and the bump class.

| lab | work | issue |
|---|---|---|
| 01 | Purge and embargo the training split | file it as part of the lab |
| 02 | LightGBM into the backtest loop; align the horizon lists | #16 |
| 03 | Activate `SNAPSHOT_S3_*` and drain the snapshot bucket for real | #43 acceptance |
| 04 | The `forecast_realizations` ledger | #13 |

Lab 01 before Lab 02: `splits.py` is already correct, so the walk-forward
comparison does not depend on the fix, but the `model_metadata` number you would
quote does.

Lab 03 involves a credential. The brief hands you the `.env` line with a
placeholder and you fill in both sides. No tooling in this repo reads, writes, or
echoes that value.

## Authoring notes

- `wowfc learn validate` after every edit. It is faster than pytest and reports
  the question id.
- Anchors must be single-line. A multi-line anchor cannot match reliably, because
  the file on disk may use CRLF while the TOML literal uses LF.
- Prefer a docstring sentence as the anchor over a code line. Prose moves less
  often than code.
- Wrong `mcq` options need a `note` saying why they are wrong. This is enforced.
- No em dashes in authored content, per the repo prose style.
