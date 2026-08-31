#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure comparison/metrics for the register-allocator study.

`AllocResult` holds one allocator's outcome on one function; `format_table`
renders a set of results. Kept binding-free so the metric arithmetic and
formatting are unit-testable; the runner (scripts/ilp_regalloc_compare.py)
builds the MachineFunctions and populates these.
"""

from dataclasses import dataclass


@dataclass
class AllocResult:
    name: str
    valid: bool
    spills: list  # list[int] spilled vreg ids
    weight: dict  # vreg -> scaled spill weight (for weighted cost)
    copies_remaining: int
    wall_time_s: float | None  # None for non-ILP baselines
    gap: float | None  # None for non-ILP baselines / unsolved
    error: str | None = None  # set when the allocator hard-failed (raised)

    @property
    def num_spills(self):
        return len(self.spills)

    @property
    def weighted_spill_cost(self):
        return sum(self.weight.get(v, 0) for v in self.spills)


def _fmt(value, spec="{}"):
    return "n/a" if value is None else spec.format(value)


def format_table(func_name, results):
    """Render one function's allocator comparison as a fixed-width table.

    An allocator that hard-failed (``error`` set) shows ``hard-fail`` with the
    reason instead of allocation metrics -- a first-class outcome in this study,
    since the whole-interval models refuse to spill.
    """
    header = ["allocator", "valid", "spills", "wspill", "copies", "time_s", "gap"]
    rows = [header]
    for r in results:
        if r.error is not None:
            rows.append([r.name, "hard-fail", r.error, "", "", "", ""])
            continue
        rows.append(
            [
                r.name,
                "yes" if r.valid else "NO",
                str(r.num_spills),
                str(r.weighted_spill_cost),
                str(r.copies_remaining),
                _fmt(r.wall_time_s, "{:.3f}"),
                _fmt(r.gap, "{:.3f}"),
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [f"== {func_name} =="]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)
