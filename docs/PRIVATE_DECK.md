# Private deck — operator discipline

The `--include-private` flag on `scripts/build_deck.py` produces a deck
variant that contains measured NPU silicon anchor values from
`.streamlit/secrets.toml`. The resulting file is named
`keyhole_results_PRIVATE.pptx` (suffix is intentional). This doc is the
operator-discipline reference for handling that file.

---

## What changes vs the public deck

| Surface | Public build (`build_deck.py`) | Private build (`build_deck.py --include-private`) |
|---|---|---|
| Output filename | `data/output/keyhole_results.pptx` | `data/output/keyhole_results_PRIVATE.pptx` |
| Slide count | 65 (current) | 66 — adds `slide_measured_silicon_anchors` at the end |
| Measured-anchor surface | absent | LLM table (3 tier-precisions × 3 models) + CNN table (2 tier-precisions × 3 CNN variants) with 🟢/🟡/🟠 source badges and value cells filled from secrets at build-time |
| What Claude sees during build | slide names in stdout only | same (build script's stdout prints no values; the loader has no print statements) |
| Where the .pptx may go | my-stuff, gdrive, any sharing surface | **Only** NXP-internal-only destinations (Kyle's laptop, NXP-internal SharePoint, etc.). Never my-stuff. Never gdrive. |

---

## The value-protection model

**Code is chat-safe.** Claude wrote `scripts/build_deck.py` +
`src/anchors/private_anchors.py` + `slide_measured_silicon_anchors` and
never saw any measurement value. The slide function references the
schema (via `LLM_TIERS`, `LLM_MODELS`, `CNN_TIERS`, `CNN_KEYS`) but the
actual `.tokps` / `.ms_per_inference` fields are populated at runtime
by the loader.

**Runtime output is private.** When Kyle runs the build, `tomllib`
reads `.streamlit/secrets.toml` (gitignored) and `python-pptx` writes
values into the .pptx directly. The .pptx now contains the values.
Kyle sees them when he opens the file. Claude does not.

**The discipline that makes this hold:**

1. **The loader (`src/anchors/private_anchors.py`) never prints values.**
   No `print(anchor.tokps)`. The fallback error message references the
   FILE PATH, not the parse error context (which could include value
   snippets from TOML). Exception messages stay value-free.
2. **`scripts/build_deck.py` stdout prints slide names only.** Running
   `python scripts/build_deck.py --include-private` produces output
   like `Building: Measured silicon anchors (LLM + CNN)` — no values.
3. **Claude must not open the resulting .pptx.** `wc -l`, `ls -la`,
   `stat` are all fine (size + filename only). `unzip`, extracting
   text, OCR on screenshots, etc. would expose values — must not
   happen.
4. **Kyle must not paste screenshots of the private slide into chat.**
   Same discipline that applies to PAI sizer / keyhole-sizer Streamlit
   apps. The values stay in Kyle's local view + the destination he
   chooses.

---

## Destination rules for `keyhole_results_PRIVATE.pptx`

| Destination | Allowed? | Why |
|---|---|---|
| Kyle's laptop (local) | ✅ Yes | The file's home |
| `kylefoxaustin/my-stuff` (GitHub private repo, but shared with collaborators / Claude sessions) | ❌ NO | This repo is the public-deck distribution channel; values would be readable by any collaborator |
| `gdrive:skippy_files/keyhole/` (Drive folder, shared) | ❌ NO | Same — values would be readable by Drive-share members |
| NXP-internal SharePoint / private wiki / email to NXP reviewers | ✅ Yes | NXP-internal-only destinations |
| Any cloud storage marked "public" / shared with non-NXP-internal parties | ❌ NO | Trivially leaks |
| Bus messages quoting values from the slide | ❌ NO | Same discipline rule — refer by KEY, never VALUE |

The `_PRIVATE` filename suffix is a visual cue, not a security boundary
— Kyle's discipline at the destination-selection step is what makes
this work.

---

## What goes into `.streamlit/secrets.toml`

The file is **gitignored** (`.gitignore:34`). The committed schema
reference is `.streamlit/secrets.toml.example` — copy it, populate
with real values locally, never commit the populated file.

Schema reference: see `.streamlit/secrets.toml.example` for the full
15-cell template. Cross-project spec at
`personal-ai-framework/docs/private_anchor_secrets_spec.md` (commit
`65bf89c` on personal-ai-framework main).

---

## What Claude can and cannot contribute to private-deck analysis

This is the work-split that Kyle and Claude agreed on when adopting B2:

**Claude can:**
- Write the loader, schema, build integration, slide structure, badge
  rendering, framework text
- Analyze projected numbers (from the rest of the deck) and write
  conclusions grounded in projections
- Author "sensitivity-band" or "bounds-based" framework text (e.g.,
  "if measured value falls in 30-50 tok/s, projection methodology is
  validated within ±15%")
- Receive bounds-language feedback from Kyle ("within the band" /
  "out of band, suggests Y") and update framework accordingly
- Author templated placeholders like *"Comparison to projection: [Kyle
  to fill in]"* where a magnitude-dependent conclusion belongs

**Claude cannot:**
- See specific measurement values
- Write conclusions like "the measured Mid INT8 number was 1.7× faster
  than my projection, so the methodology underestimates by X%"
- Update quantitative claims in the briefing based on measured data
- Read the resulting `keyhole_results_PRIVATE.pptx` to verify content

**Kyle owns:** the magnitude-dependent conclusion paragraphs that
require seeing the measured values. These can be hand-written into the
private deck as textbox annotations, or left as templated placeholders
for Kyle to fill in after build.

---

## Verifying the discipline is intact

After `git pull` or any code review:

```bash
# 1. Loader has no value-printing calls (should return no matches in privacy-sensitive contexts)
grep -n "print.*tokps\|print.*ms_per_inference" src/anchors/private_anchors.py

# 2. Build script in private mode prints slide names only (sample run)
echo "" > /tmp/test_secrets.toml  # empty secrets — proves the no-values default state
python scripts/build_deck.py --include-private | grep -iE "tokps|ms_per_inference|tok/s|ms\b" || echo "OK: no values in stdout"

# 3. The gitignored secrets file is not in git
git check-ignore -v .streamlit/secrets.toml  # should report it's ignored
```
