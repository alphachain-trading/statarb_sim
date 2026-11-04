# Architecture decision records

An ADR records one decision that shapes more than one module or the overall
architecture: the context, the decision and its consequences. ADRs explain why the
code is the way it is; status still lives on GitHub (see
[CONTRIBUTING.md](../../CONTRIBUTING.md)).

## Numbering

Files are named `NNNN-short-slug.md`, numbered sequentially from `0001`. Numbers are
never reused. Copy [`0000-template.md`](0000-template.md) to start a new one.

An accepted ADR is not edited in substance. To change a decision, write a new ADR and
set the old one's status to "Superseded by NNNN".

## ADR or issue comment?

- **ADR**: the decision affects more than one module or the overall architecture, e.g.
  artifact schemas, identity keys, the date model, where a policy lives.
- **Issue comment**: the decision concerns a single issue (approach, intended
  behaviour, scope, discard, priority), as described in CONTRIBUTING.md.

If in doubt, start with an issue comment; promote it to an ADR when a second module
depends on it.
