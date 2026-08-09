# ClinicalTrials.gov API v2 — Verified Findings

Compiled and live-verified on 2026-08-08 while prepping for the Cheiron take-home
(ClinicalTrials.gov Query-to-Visualization Agent). Where this document disagrees
with third-party docs/tutorials, trust this — it was confirmed against the live
API, and several widely-repeated secondary sources turned out to be wrong.

## Basics

- **Base URL:** `https://clinicaltrials.gov/api/v2`
- **Auth:** none required — public API
- **Rate limit:** ~50 requests/minute per IP (not hit during testing, but plan
  for basic retry/backoff on 429)
- **Formats:** JSON (default) or CSV via `format=csv`
- **Live version check:** `GET /api/v2/version` → confirmed working, returns:
  ```json
  { "apiVersion": "2.0.5", "dataTimestamp": "2026-08-07T09:00:05" }
  ```
  Useful for citing "data as of" in the README.

## Core endpoints

| Endpoint | Purpose |
|---|---|
| `GET /studies` | Search — primary endpoint for everything in this assignment |
| `GET /studies/{nctId}` | Full record for one trial |
| `GET /studies/metadata` | Field metadata/schema |
| `GET /stats/field/values` | Server-side aggregation — **returned 500 in testing, avoid** (also can't support per-record citations even if it worked, since data never passes through our own records) |
| `GET /version` | API version + data freshness timestamp — confirmed working |

## Search parameters — VERIFIED against live API

### Confirmed working
- `query.cond` — condition/disease (e.g. `lung cancer`)
- `query.intr` — drug/intervention (e.g. `Pembrolizumab`)
- `query.spons` — sponsor
- `query.locn` — location
- `query.term` — free-text / Essie expression search
- `fields` — comma-separated, **PascalCase** names (e.g. `NCTId,BriefTitle,Phase`) — restricts response size, always use this
- `pageSize` — max 1000, default is only 10 if omitted — always set explicitly
- `pageToken` — pass the `nextPageToken` from the previous response for the next page. **No offset-based pagination exists.**
- `countTotal=true` — include on first request to get `totalCount` in one call

### ✅ Combining several values in one `query.*` parameter — verified live

`OR` is a true set **union**; a comma is an **intersection**. Both are silent —
neither errors, so the only way to tell them apart is to count.

| request | `totalCount` |
| --- | --- |
| `query.intr=Pembrolizumab` | 2,922 |
| `query.intr=Nivolumab` | 2,016 |
| `query.intr=Pembrolizumab OR Nivolumab` | **4,648** |
| `query.intr=Pembrolizumab, Nivolumab` | 290 |

Inclusion–exclusion closes exactly: `2922 + 2016 − 290 = 4648`. The same holds
for `query.cond` (`3743 + 14425 − 652 = 17516`).

This is why `build_searches` joins multiple extracted values with `" OR "`
rather than keeping only the first. The comma form is deliberately never
emitted: it looks like a list and behaves like an `AND`, which is the kind of
silent narrowing this file exists to record.

### ✅ Multiple values inside ONE `aggFilters` key — verified live 2026-08-09

Values are **space-separated** and union. The comma separates *different keys*,
not values of the same key — repeating a key silently keeps only one of them.

| request | `totalCount` | |
| --- | --- | --- |
| `status:rec` | 64,847 | |
| `status:com` | 326,301 | |
| `status:rec com` | **391,148** | = 64,847 + 326,301 exactly (statuses are disjoint) |
| `status:rec com ter` | **425,230** | = + 34,082 exactly |
| `status:rec,status:com` | 326,301 | ⚠️ repeated key — one value silently wins |
| `status:rec\|com` | 0 | ⚠️ silent failure |
| `status:rec OR status:com` | HTTP 400 | loud failure |

Same for phase, where the union is visibly *less* than the sum because a trial
can hold several phases:

| request | `totalCount` |
| --- | --- |
| `phase:2` | 89,652 |
| `phase:3` | 49,614 |
| `phase:2 3` | **131,704** (< 139,266 — combined Phase 2/3 trials counted once) |

Sampling the union confirms a genuine mix (`PHASE2`, `PHASE3`, `PHASE1/PHASE2`,
`PHASE2/PHASE3`), not one filter winning.

### ✅ `aggFilters` status codes — nine verified, two are traps

Each verified by checking the filtered sample contains **only** that status:

| code | status | count |
| --- | --- | --- |
| `rec` | RECRUITING | 64,847 |
| `com` | COMPLETED | 326,301 |
| `act` | ACTIVE_NOT_RECRUITING | 21,858 |
| `not` | NOT_YET_RECRUITING | 29,086 |
| `enr` | ENROLLING_BY_INVITATION | 5,236 |
| `ter` | TERMINATED | 34,082 |
| `sus` | SUSPENDED | 1,750 |
| `wit` | WITHDRAWN | 16,627 |
| `unk` | UNKNOWN | 95,858 |

⚠️ `status:avail` and `status:no_lon` both return **HTTP 200 with zero results**
— the same silent-failure shape as `phase:PHASE3`. AVAILABLE and
NO_LONGER_AVAILABLE are therefore filtered client-side.

### ⚠️ DOES NOT EXIST — confirmed 400 error live
- `filter.phase=PHASE3` → **`{"error": "unknown parameter"}` (HTTP 400)**
  Multiple third-party doc sources (GitHub reference docs, dev.to tutorials,
  even some MCP server implementations) confidently document this as valid.
  It is not, as of API v2.0.5 (2026-08-08). Do not trust this parameter name.

### ✅ CORRECT phase/status filtering — `aggFilters`
Phase and status filtering actually go through a single combined parameter:

```
aggFilters=phase:3
```

**Critical detail:** the phase value is a **bare number** (`3`), not the full
enum string (`PHASE3`). `aggFilters=phase:PHASE3` returns HTTP 200 but with
zero results (silently wrong, not an error) — worth guarding against in code
with a sanity check, since it fails silently rather than loudly.

Combine multiple filters with commas:
```
aggFilters=phase:3,status:rec
```
Status values in `aggFilters` appear to use short/abbreviated codes (e.g.
`status:rec` for recruiting, `status:com` for completed) — **only
`status:rec` was directly confirmed live during this session; treat other
status abbreviations as unverified until tested.**

**Still unverified / worth confirming before relying on it:** whether
`filter.overallStatus` (a separate dot-namespaced param, distinct from
`aggFilters`) is still valid in this API version, or whether all `filter.*`
namespaced params were folded into `aggFilters`. Test with:
```bash
curl -s "https://clinicaltrials.gov/api/v2/studies?query.intr=Pembrolizumab&pageSize=2&filter.overallStatus=RECRUITING"
```
before depending on it in the fetcher.

## Response shape — confirmed live

```json
{
  "studies": [
    {
      "protocolSection": {
        "identificationModule": {
          "nctId": "NCT06415487",
          "briefTitle": "ACE2016 in Adult Subjects With Locally Advanced or Metastatic Solid Tumors Expressing EGFR"
        },
        "statusModule": {
          "overallStatus": "RECRUITING"
        },
        "sponsorCollaboratorsModule": {
          "leadSponsor": { "name": "Acepodia Biotech, Inc." }
        },
        "designModule": {
          "phases": ["PHASE1"],
          "enrollmentInfo": { "count": 30 }
        }
      }
    }
  ],
  "nextPageToken": "ZVdj7o2Elu8o3lp2Vcm44unumpOQJJxnbPip"
}
```

Key structural notes:
- Response JSON keys are **camelCase**, nested under
  `protocolSection.<moduleName>Module.<field>` — this does NOT map 1:1 to the
  PascalCase names used in the `fields` request parameter (e.g. request
  `Phase` → response `designModule.phases`). Confirm exact request-name →
  response-path mapping per field as you build the fetcher, don't assume a
  simple case conversion.
- **`phases` is always an array**, even for a single-phase trial
  (`"phases": ["PHASE1"]`). Some trials span multiple phases (e.g. combined
  Phase 1/2 studies) — aggregation logic must handle multi-element arrays,
  not assume exactly one phase per trial.
- No `totalCount` field appears in the example above because `countTotal=true`
  wasn't passed on that request — include it explicitly when you need totals.

## Known data quality issues (from secondary research, not yet independently reverified live)

- Roughly a quarter of all studies have no usable phase value — either the
  enum `NA` (usually observational studies, where phase doesn't apply) or the
  field is absent entirely. Aggregators grouping by phase should explicitly
  report/bucket "no phase data" rather than silently dropping those trials or
  miscounting.
- Date fields are inconsistently granular across records (some
  year-precision only, some full ISO dates) — don't assume every trial has a
  clean parseable full date for time-series bucketing.

## Debugging notes / gotchas hit during this session

1. **Shell quoting matters.** Unquoted URLs with `&` in bash/zsh get
   interpreted as command backgrounding — always wrap the full URL in double
   quotes when testing with curl.
2. **`curl -v` shows headers, not the response body**, when the request
   fails — use plain `curl -s` (no `-v`) to see the actual JSON/text error
   message body when debugging a 400.
3. **A 200 response is not proof your filter did what you think.**
   `aggFilters=phase:PHASE3` returned HTTP 200 with an empty `studies: []`
   array — indistinguishable from "no matching trials" unless you already
   know Pembrolizumab has many Phase 3 trials. Sanity-check filter results
   against a query you know should return non-trivial results, not just
   check the status code.

## Sanity-check commands used to verify the above

```bash
# Version/freshness check
curl -s "https://clinicaltrials.gov/api/v2/version"

# Confirms filter.phase does not exist (400)
curl -s "https://clinicaltrials.gov/api/v2/studies?query.intr=Pembrolizumab&pageSize=2&filter.phase=PHASE3"

# Confirms fields + basic search works (200, real data)
curl -s "https://clinicaltrials.gov/api/v2/studies?query.intr=Pembrolizumab&pageSize=2&fields=NCTId,BriefTitle,OverallStatus,Phase,LeadSponsorName,EnrollmentCount"

# Confirms aggFilters=phase:PHASE3 is silently wrong (200, empty results)
curl -s "https://clinicaltrials.gov/api/v2/studies?query.intr=Pembrolizumab&pageSize=2&aggFilters=phase:PHASE3"

# Confirms aggFilters=phase:3 (bare number) is correct
curl -s "https://clinicaltrials.gov/api/v2/studies?query.intr=Pembrolizumab&pageSize=2&aggFilters=phase:3"

# Confirms OR is a union and comma is an intersection (compare the totalCounts:
# 2922, 2016, 4648, 290 -- and 2922 + 2016 - 290 == 4648)
for q in "Pembrolizumab" "Nivolumab" "Pembrolizumab OR Nivolumab" "Pembrolizumab, Nivolumab"; do
  curl -s -G "https://clinicaltrials.gov/api/v2/studies" \
    --data-urlencode "query.intr=$q" -d countTotal=true -d pageSize=1 |
    python3 -c "import json,sys; print(json.load(sys.stdin)['totalCount'])"
done
```

## Open items to verify during implementation (not yet tested live)

- [ ] `filter.overallStatus` as a standalone param — still valid, or folded into `aggFilters`?
- [ ] Full list of valid `aggFilters` keys beyond `phase` and `status` (e.g. `studyType`, `ages`)
- [ ] Exact abbreviated status codes accepted by `aggFilters=status:*` beyond `rec`
- [ ] Whether `armsInterventionsModule` reliably contains enough text to build citation excerpts without a separate per-study `/studies/{nctId}` call
