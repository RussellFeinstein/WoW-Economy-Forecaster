# M02. API to rows

Part I. The domain and the data. Prereq: M01. No lab.

## Why this module exists

Ingestion is the layer everyone skims. It reads like plumbing: get a token, make a
request, insert some rows. But every number this system ever reports is a
transformation of what happens here, and four decisions made in these three files
propagate all the way to a recommendation.

Which price field wins. What a row represents. What `observed_at` measures. What
counts as a successful run.

Get any of those wrong in your head and you will misread a chart three modules
later and blame the model.

## The idea to hold onto

A raw observation is one auction listing at one moment, as the client saw it:

```
one API call        -> ~314,000 auction listings
one listing         -> one row in market_observations_raw
min_buyout_raw      -> that listing's unit price, not a minimum of anything
observed_at         -> the client's clock, not Blizzard's
```

No aggregation happens at ingest. That is deliberate: the API serves only the
current snapshot, so anything you fail to store is gone forever, and anything you
aggregate too early you cannot un-aggregate. The raw table keeps the full record on
every row and lets the daily rollup decide what a price means.

The price of that choice is size, and the bound on it is the 30-day Terms of
Service retention wall. Raw rows are a working set. The rollups are the history.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/ingestion/blizzard_client.py`](../../wow_forecaster/ingestion/blizzard_client.py)
  The module docstring documents the OAuth2 flow and both AH endpoints. Then read
  `_parse_commodities_response` and notice how much it hard-codes: `buyout=0`,
  `bid=0`, `realm_slug=self.region`, `time_left="VERY_LONG"`, and `fetched_at` set
  from the local clock.
- [`wow_forecaster/ingestion/snapshot.py`](../../wow_forecaster/ingestion/snapshot.py)
  Small and worth reading in full. The envelope shape, the deterministic path
  builder, and `compute_hash`. Ask yourself what exactly the hash covers.
- [`wow_forecaster/pipeline/ingest.py`](../../wow_forecaster/pipeline/ingest.py)
  The centre of the module. Read `parse_blizzard_records` for the price fallback
  chain, then read `_execute` for the three-phase connection pattern, following the
  phase comments. Then ask what the stage returns when nothing was inserted.
- [`wow_forecaster/models/market.py`](../../wow_forecaster/models/market.py)
  `RawMarketObservation` versus `NormalizedMarketObservation`. Compare the types of
  `min_buyout_raw` and `price_gold`, and work out where a null goes.
- [`wow_forecaster/db/schema.py`](../../wow_forecaster/db/schema.py)
  The `market_observations_raw` DDL. Note what constraints are there and, more
  usefully, which one is not.

Follow one thread outward if you have time:
[`wow_forecaster/ingestion/cloud_sync.py`](../../wow_forecaster/ingestion/cloud_sync.py)
documents the selection rules for the catch-up path, and rule 4 is the clearest
statement in the repo of what `observed_at` does and does not mean.

## What you should be able to do afterwards

- Trace a commodity snapshot from the token request to an inserted row, naming each
  hop and what it produces.
- State the `min_buyout_raw` fallback chain and say what unit each branch is in.
- Explain the three-phase connection pattern and the exact failure it prevents.
- Say why the item foreign-key guard is loaded before the fetch rather than checked
  at insert time.
- Describe what dedups a raw observation. The answer is shorter than you expect.
- Predict what a fully-skipped run looks like from the outside.

## A note on what this layer does not guarantee

Two things in this module are worth remembering as limits rather than facts.

The raw table has no identity concept. There is no unique constraint, no
`ON CONFLICT`, and `observed_at` is a client timestamp, so two captures of the same
hourly dump are simply two sets of rows. Idempotency here is a property of the
schedule, not of the code.

And a run that skipped every record reports the same status as a run that inserted
everything. That is not a bug in any single function: the snapshot succeeded, the
stage returned, the base class stamped success. It is a gap between what the code
measures and what you care about, and this project has already paid for that exact
gap once. `check-data-health` exists because a status reported by the thing that
failed is not evidence.

M03 picks up where this stops: turning these irregular per-listing rows into a
dense daily series.
