# Knowledge-sharing articles

Follow the **Knowledge sharing articles** section of `../CLAUDE.md` for the site
workflow and visual rules.

- Open by giving the reader the intuition needed to understand the problem.
  Use a concrete failure mode or decision to make that intuition tangible,
  then state the repo's design claim once the mechanism is clear.
- Build the explanation in causal order, from general mechanism to this repo's
  implementation. Each section should give the reader one idea needed by the
  next. Close with a compact recap of terms that matter to the argument.
- Write connected engineering prose. Ground claims in this repo's code and explain
  why each design choice produces the stated behavior. Use a measured, senior
  architect's voice: precise, structured, and free of casual asides or slogans.
  State the central rule once, then spend later sections on its consequences
  instead of repeating it. Replace generic definitions with a specific
  operation and the consequence of changing it. Keep limitations precise and
  proportionate to the article's main point.
- Use only the diagrams needed to explain the argument. Give each SVG clear alt
  text and a caption. Check every arrow, seed source, process boundary, and time
  boundary against the code; a diagram must not imply that separate seeds are
  derived from one another. Preserve the site's visual style and register every
  asset in `pages/hooks.py`.
- Build the site and inspect the published page at desktop and narrow widths.
