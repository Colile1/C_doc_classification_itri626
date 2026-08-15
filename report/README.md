# report/

The written report. Approximately 10 to 15 pages
excluding references and appendices, submitted as PDF.

## Planned files

| File | Responsibility |
|------|----------------|
| `report.md` | The working draft. Prose is written under the honours writing rules, not generated in one pass |
| `references.bib` | Sources: RVL-CDIP, the three architecture papers, the document-classification prior work, the libraries |
| `figures/` | Copies of the specific figures from `outputs/figures/` that the report actually uses, so the document does not break when a run is re-executed |

## Integration

Every claim in the results section must name the run that produced it.
The figures here are copies, not the originals: `outputs/` is
regenerable, `report/figures/` is frozen at the version that was cited.

Every number in the prose must be present in `outputs/metrics/`.
