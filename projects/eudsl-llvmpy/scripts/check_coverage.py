#!/usr/bin/env python3
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compute overall line coverage for eudsl-llvmpy C++ sources and enforce a threshold.

Usage (with pre-generated LCOV file):
    check_coverage.py --lcov <path> --sources <dir>... [--threshold=97]

Usage (with llvm-cov, generates LCOV internally):
    check_coverage.py --llvm-cov <path> --profdata <path> --objects <obj>...
                      --sources <dir>... [--threshold=97]

Handles LCOV_EXCL_LINE, LCOV_EXCL_START, and LCOV_EXCL_STOP filtering by
reading source files directly.

Exits 0 if coverage >= threshold, 1 otherwise.
"""

import re
import subprocess
import sys
import argparse
from pathlib import Path


def parse_lcov(lcov_text):
    """Parse LCOV text into per-file line hit data.

    Returns {filepath: {lineno: hit_count, ...}, ...}
    """
    file_hits = {}
    current_file = None

    for line in lcov_text.splitlines():
        if line.startswith("SF:"):
            current_file = line[3:]
            if current_file not in file_hits:
                file_hits[current_file] = {}
        elif line.startswith("DA:"):
            if current_file is None:
                continue
            parts = line[3:].split(",", 2)
            if len(parts) >= 2:
                try:
                    lineno = int(parts[0])
                    count = int(parts[1])
                    file_hits[current_file][lineno] = (
                        file_hits[current_file].get(lineno, 0) + count
                    )
                except ValueError:
                    pass
        elif line == "end_of_record":
            current_file = None

    return file_hits


def parse_lcov_functions(lcov_text):
    """Parse LCOV FN/FNDA records into per-file, per-function data.

    Line coverage alone misses an accessor lambda whose body shares a source
    line with the enclosing `.def(...)` registration call: the registration
    runs at import, so the line is marked covered even though the lambda body
    never executes. Each lambda is its own FUNCTION in the coverage data, so
    function coverage (FNDA execution counts) catches these. The X-macro
    dispatch (e.g. valueTypeInfo) stays a single function, so this does not
    penalize its many unreachable case regions.

    Returns {filepath: {name: {"line": lineno, "count": hits}, ...}, ...}
    """
    file_fns = {}
    current_file = None

    for line in lcov_text.splitlines():
        if line.startswith("SF:"):
            current_file = line[3:]
            file_fns.setdefault(current_file, {})
        elif line.startswith("FN:"):
            if current_file is None:
                continue
            # FN:<line>,<name>  (llvm-cov) or FN:<start>,<end>,<name>
            rest = line[3:]
            head, _, name = rest.rpartition(",")
            first = head.split(",", 1)[0]
            try:
                lineno = int(first)
            except ValueError:
                continue
            entry = file_fns[current_file].setdefault(name, {"line": None, "count": 0})
            entry["line"] = lineno
        elif line.startswith("FNDA:"):
            if current_file is None:
                continue
            count_str, _, name = line[5:].partition(",")
            try:
                count = int(count_str)
            except ValueError:
                continue
            entry = file_fns[current_file].setdefault(name, {"line": None, "count": 0})
            entry["count"] = count
        elif line == "end_of_record":
            current_file = None

    return file_fns


def demangle(names):
    """Best-effort demangling for nicer reporting. Returns {name: pretty}."""
    stripped = [n.split(":", 1)[-1] for n in names]  # drop any "file:" prefix
    for tool in ("llvm-cxxfilt", "c++filt"):
        try:
            out = subprocess.run(
                [tool], input="\n".join(stripped), capture_output=True, text=True
            )
            if out.returncode == 0:
                pretty = out.stdout.splitlines()
                if len(pretty) == len(names):
                    return dict(zip(names, pretty))
        except (OSError, FileNotFoundError):
            continue
    return {n: s for n, s in zip(names, stripped)}


def get_excluded_lines(filepath):
    """Read a source file and return the set of line numbers excluded by
    LCOV_EXCL_LINE, LCOV_EXCL_START, and LCOV_EXCL_STOP markers.
    """
    excluded = set()
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return excluded

    in_exclusion_block = False
    for i, line in enumerate(lines, start=1):
        if "LCOV_EXCL_START" in line:
            in_exclusion_block = True
            excluded.add(i)
        elif "LCOV_EXCL_STOP" in line:
            in_exclusion_block = False
            excluded.add(i)
        elif in_exclusion_block:
            excluded.add(i)
        elif "LCOV_EXCL_LINE" in line:
            excluded.add(i)

    return excluded


def file_in_sources(filepath, source_dirs):
    """Check if filepath is under one of the source directories."""
    fp = Path(filepath).resolve()
    for src in source_dirs:
        sp = Path(src).resolve()
        try:
            fp.relative_to(sp)
            return True
        except ValueError:
            pass
    return False


def generate_lcov(llvm_cov, profdata, objects):
    """Run llvm-cov export to produce LCOV text."""
    cmd = [llvm_cov, "export", "-format=lcov"]
    for obj in objects:
        cmd += ["-object", obj]
    cmd += [f"-instr-profile={profdata}"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"llvm-cov export failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Check eudsl-llvmpy C++ source coverage")
    parser.add_argument("--lcov", default=None, help="Path to LCOV .info file")
    parser.add_argument("--llvm-cov", default=None, help="Path to llvm-cov binary")
    parser.add_argument(
        "--profdata", default=None, help="Path to merged .profdata file"
    )
    parser.add_argument(
        "--objects", default=None, nargs="+", help="Instrumented object files"
    )
    parser.add_argument(
        "--sources", required=True, nargs="+", help="Source directories to include"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=97.0,
        help="Minimum required line coverage percentage (default: 97)",
    )
    parser.add_argument(
        "--function-threshold",
        type=float,
        default=None,
        help="Minimum required function coverage percentage "
        "(default: same as --threshold). Function coverage catches accessor "
        "lambdas whose body never runs even though their registration line is "
        "covered.",
    )
    parser.add_argument(
        "--ignore-filename-regex",
        default=None,
        help="Skip files matching this regex",
    )
    args = parser.parse_args()

    if args.lcov:
        with open(args.lcov) as f:
            lcov_text = f.read()
    elif args.llvm_cov and args.profdata and args.objects:
        lcov_text = generate_lcov(args.llvm_cov, args.profdata, args.objects)
    else:
        parser.error("Either --lcov or (--llvm-cov, --profdata, --objects) is required")

    ignore_re = re.compile(args.ignore_filename_regex) if args.ignore_filename_regex else None

    file_hits = parse_lcov(lcov_text)
    file_fns = parse_lcov_functions(lcov_text)

    file_stats = {}
    for filepath, line_counts in file_hits.items():
        if not file_in_sources(filepath, args.sources):
            continue
        if ignore_re and ignore_re.search(filepath):
            continue

        excluded = get_excluded_lines(filepath)

        total = 0
        covered = 0
        missed = []

        for lineno, count in sorted(line_counts.items()):
            if lineno in excluded:
                continue
            total += 1
            if count > 0:
                covered += 1
            else:
                missed.append(lineno)

        # Function coverage: a function whose definition line is excluded
        # (e.g. an LCOV_EXCL_START/STOP block like the fatal handler) is
        # dropped, mirroring the line-coverage exclusion.
        f_total = 0
        f_covered = 0
        f_missed = []
        for name, info in file_fns.get(filepath, {}).items():
            ln = info["line"]
            if ln is not None and ln in excluded:
                continue
            f_total += 1
            if info["count"] > 0:
                f_covered += 1
            else:
                f_missed.append((ln if ln is not None else 0, name))

        file_stats[filepath] = {
            "total": total,
            "covered": covered,
            "missed": missed,
            "f_total": f_total,
            "f_covered": f_covered,
            "f_missed": f_missed,
        }

    total_lines = sum(s["total"] for s in file_stats.values())
    covered_lines = sum(s["covered"] for s in file_stats.values())
    percent = (covered_lines / total_lines * 100.0) if total_lines > 0 else 0.0

    total_fns = sum(s["f_total"] for s in file_stats.values())
    covered_fns = sum(s["f_covered"] for s in file_stats.values())
    fn_percent = (covered_fns / total_fns * 100.0) if total_fns > 0 else 100.0

    fn_threshold = (
        args.function_threshold
        if args.function_threshold is not None
        else args.threshold
    )

    print(f"eudsl-llvmpy C++ coverage: {covered_lines}/{total_lines} lines ({percent:.2f}%)")
    print(f"eudsl-llvmpy C++ coverage: {covered_fns}/{total_fns} functions ({fn_percent:.2f}%)")

    # Demangle any missed function names for readable reporting.
    all_missed = [name for s in file_stats.values() for _, name in s["f_missed"]]
    pretty = demangle(all_missed) if all_missed else {}

    for filename, stats in sorted(file_stats.items()):
        f_total = stats["total"]
        f_covered = stats["covered"]
        f_percent = (f_covered / f_total * 100.0) if f_total > 0 else 0.0
        fn_t = stats["f_total"]
        fn_c = stats["f_covered"]
        fn_p = (fn_c / fn_t * 100.0) if fn_t > 0 else 100.0
        print(
            f"  {filename}: {f_covered}/{f_total} lines ({f_percent:.2f}%), "
            f"{fn_c}/{fn_t} functions ({fn_p:.2f}%)"
        )
        if stats["missed"]:
            sorted_lines = sorted(stats["missed"])
            ranges = []
            start = prev = sorted_lines[0]
            for l in sorted_lines[1:]:
                if l == prev + 1:
                    prev = l
                else:
                    ranges.append(f"{start}-{prev}" if prev > start else str(start))
                    start = prev = l
            ranges.append(f"{start}-{prev}" if prev > start else str(start))
            print(f"    missed lines: {', '.join(ranges)}")
        for ln, name in sorted(stats["f_missed"]):
            print(f"    missed function (line {ln}): {pretty.get(name, name)}")

    line_ok = percent >= args.threshold
    fn_ok = fn_percent >= fn_threshold
    if not line_ok or not fn_ok:
        if not line_ok:
            print(f"\nFAILED: line coverage {percent:.2f}% < threshold {args.threshold:.2f}%")
        if not fn_ok:
            print(
                f"\nFAILED: function coverage {fn_percent:.2f}% < threshold "
                f"{fn_threshold:.2f}% (an accessor/binding runs at registration "
                f"but its body is never called by a test)"
            )
        return 1

    print(
        f"\nPASSED: line coverage {percent:.2f}% >= {args.threshold:.2f}%, "
        f"function coverage {fn_percent:.2f}% >= {fn_threshold:.2f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
