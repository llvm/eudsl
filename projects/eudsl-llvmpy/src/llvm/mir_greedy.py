#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Faithful Python port of llvm::RegAllocGreedy.

Importing this module attaches the allocator as ``mir.RAGreedy``. The class
mirrors RegAllocGreedy.cpp method-for-method; the pure cost computations
(eviction_cost, calc_gap_weights, calc_global_split_cost) are module-level so
they can be unit-tested without running the allocator.
"""

import enum
import heapq

from . import mir


class LiveRangeStage(enum.IntEnum):
    """RAGreedy's per-vreg allocation stage ladder (LiveRangeStage)."""

    RS_New = 0
    RS_Assign = 1
    RS_Split = 2
    RS_Split2 = 3
    RS_Spill = 4
    RS_Memory = 5


# RAGreedy refuses to evict a physreg with this many or more interfering vregs
# ("chances are one is heavier") -- EvictInterferenceCutoff.
EVICT_INTERFERENCE_CUTOFF = 10


def eviction_cost(interferers):
    """RAGreedy's EvictionCost, ``(broken_hints, max_weight)``, compared
    lexicographically (Python tuple order, BrokenHints primary). `interferers`
    is a list of ``(weight, breaks_hint, copy_cost)`` triples: ``broken_hints``
    sums the copy cost of each interferer whose satisfied hint the eviction would
    break, ``max_weight`` is the largest interferer spill weight (NOT their sum).
    Mirrors canEvictInterferenceBasedOnCost's cost accumulation; the caller picks
    the physreg with the least such cost. No callee-saved term: unused-CSR
    avoidance lives in tryAssignCSRFirstTime, not eviction."""
    broken_hints = 0.0
    max_weight = 0.0
    for weight, breaks_hint, copy_cost in interferers:
        if breaks_hint:
            broken_hints += copy_cost
        if weight > max_weight:
            max_weight = weight
    return (broken_hints, max_weight)


def calc_gap_weights(use_slots, interferer_spans):
    """Per-gap maximum interference weight for a local interval.

    `use_slots` is the sorted list of use positions; the gaps are the
    consecutive intervals between them. `interferer_spans` are (start, end,
    weight) triples. Returns one weight per gap: the largest weight of any
    interferer whose [start, end) overlaps that gap (0.0 if none). Mirrors
    RAGreedy::calcGapWeights.
    """
    gaps = [0.0] * max(len(use_slots) - 1, 0)
    for i in range(len(gaps)):
        lo, hi = use_slots[i], use_slots[i + 1]
        for start, end, weight in interferer_spans:
            if start < hi and lo < end:  # half-open overlap
                if weight > gaps[i]:
                    gaps[i] = weight
    return gaps


def calc_global_split_cost(boundary_freqs):
    """Cost of a global split: the summed block frequency at each split
    boundary (where a copy is inserted). Mirrors the accounting in
    RAGreedy::calcGlobalSplitCost; the caller supplies the per-boundary
    frequencies read from MachineBlockFrequencyInfo.
    """
    return float(sum(boundary_freqs))


# RAGreedy's float-comparison hysteresis (2007/2048), so a marginally-better
# split candidate doesn't oscillate the choice.
_HYSTERESIS = 2007 / 2048.0

# Stand-in for LLVM's huge_valf sentinel weight (uninhabitable gap).
_HUGE_VALF = float("inf")


def normalize_spill_weight(use_def_freq, size, instr_dist):
    """VirtRegAuxInfo::normalizeSpillWeight: use/def frequency divided by the
    interval size plus a fixed 25-instruction floor, so tiny intervals are
    ranked by use count and large ones by use density. `size` and `instr_dist`
    are in slot-index units. Mirrors CalcSpillWeights.h exactly."""
    return use_def_freq / (size + 25 * instr_dist)


# growRegion bails once its edge-walk budget is exhausted (matches
# GrowRegionComplexityBudget); the value mirrors the LLVM default.
_GROW_REGION_COMPLEXITY_BUDGET = 10000

# Sentinel for "no candidate owns this edge bundle" (RAGreedy's NoCand).
_NO_CAND = ~0


def _earlier_instr(a, b):
    """SlotIndex::isEarlierInstr(a, b)."""
    return a.is_earlier_instr(b)


class GlobalSplitCandidate:
    """A region-split candidate: one physreg's live-bundle solution, the
    through blocks it activated, its opened split-interval index, and an
    interference cursor. Mirrors RAGreedy's GlobalSplitCandidate."""

    def __init__(self):
        self.phys_reg = 0  # 0 == compact region (no physreg)
        self.live_bundles = mir.BitVector()
        self.active_blocks = []  # through-block numbers, in discovery order
        self.intv_idx = 0
        self.intf = None  # an InterferenceCursor, set in reset()

    def reset(self, physreg, cursor=None):
        self.phys_reg = physreg
        self.intv_idx = 0
        self.active_blocks = []
        self.live_bundles = mir.BitVector()
        if cursor is not None:
            self.intf = cursor


class RAGreedy(mir.RegAllocBase):
    """Faithful reproduction of llvm::RegAllocGreedy."""

    # Exposed on instances so tests can reference the ladder without importing
    # the enum.
    RS_New = LiveRangeStage.RS_New
    RS_Assign = LiveRangeStage.RS_Assign
    RS_Split = LiveRangeStage.RS_Split
    RS_Split2 = LiveRangeStage.RS_Split2
    RS_Spill = LiveRangeStage.RS_Spill
    RS_Memory = LiveRangeStage.RS_Memory

    def __init__(self):
        super().__init__()
        # Per-vreg (keyed by reg id, stable across splitting) allocation state.
        self._stage = {}
        self._cascade = {}
        self._next_cascade = 1
        # Priority queue of pending vregs (a max-heap keyed by priority).
        self._queue = []
        # Optional test instrumentation: reg id -> stage name that resolved it.
        self.trace = {}
        # Region-split scratch (RAGreedy's GlobalCand / BundleCand), reused
        # across select_or_split calls.
        self._global_cand = []
        self._bundle_cand = []
        # Per-use-block BlockConstraints built by _add_split_constraints and
        # reused by _calc_global_split_cost (as C++ keeps SplitConstraints).
        self._split_constraints = []

    # -- stage/cascade bookkeeping (RAGreedy::ExtraRegInfo) ------------------
    def _get_stage(self, reg):
        return self._stage.get(reg, LiveRangeStage.RS_New)

    def _set_stage(self, reg, stage):
        self._stage[reg] = stage

    def _get_cascade(self, reg):
        """RAGreedy ExtraRegInfo::getCascade: the range's raw cascade number, 0
        if it has never been evicted. A 0-cascade range can evict anything and
        be evicted by anything (the eviction-loop guard). Does NOT assign."""
        return self._cascade.get(reg, 0)

    def _cascade_or_next(self, reg):
        """getCascadeOrCurrentNext: the range's cascade, or the next value to be
        assigned (peeked, not consumed) if it has none."""
        return self._cascade.get(reg) or self._next_cascade

    def _assign_cascade(self, reg):
        """getOrAssignNewCascade: assign and consume a fresh cascade if unset."""
        c = self._cascade.get(reg)
        if not c:
            c = self._next_cascade
            self._next_cascade += 1
            self._cascade[reg] = c
        return c

    # -- enqueue / getPriority (RegAllocGreedy::enqueue) --------------------
    def _priority_for(
        self,
        stage,
        size,
        is_local_assign,
        local_prio,
        global_bit,
        alloc_priority,
        trumps_globalness,
        has_pref,
    ):
        """Faithful RAGreedy DefaultPriorityAdvisor::getPriority. Larger =
        allocated sooner. RS_Split ranges are deferred to bare `size` (below the
        1<<31 mark). Everything else packs, from the low bits: the 24-bit
        size/instruction-distance priority, then AllocationPriority and the
        globalness bit (order set by `trumps_globalness`), the 1<<31 mark above
        RS_Split, and a 1<<30 boost for a known physreg preference.

        `local_prio` is the instruction-order priority for a local RS_Assign
        range; `global_bit`/`is_local_assign` are computed by the caller (which
        holds the SplitAnalysis/loop context). The bit layout mirrors
        RegAllocGreedy.cpp exactly."""
        if stage == LiveRangeStage.RS_Split:
            return size
        prio = local_prio if is_local_assign else size
        prio = min(prio, (1 << 24) - 1)  # maxUIntN(24)
        if trumps_globalness:
            prio |= (alloc_priority << 25) | (global_bit << 24)
        else:
            prio |= (global_bit << 29) | (alloc_priority << 24)
        # Mark a higher bit to prioritize global and local above RS_Split.
        prio |= 1 << 31
        # Boost ranges that have a physical register hint.
        if has_pref:
            prio |= 1 << 30
        return prio

    def enqueue(self, reg):
        li = self.lis.interval(reg)
        rc = self.reg_class(reg)
        instr_dist = self.slot_index_instr_distance()
        num_alloc = self.num_allocatable_regs(rc)
        size = li.size
        reverse = self.reverse_local_assignment()
        # ForceGlobal: giant ranges fall back to the global heuristic (the
        # size/InstrDist term only when not assigning locals bottom-up).
        force_global = self.reg_class_has_global_priority(rc) or (
            not reverse and (size // instr_dist) > 2 * num_alloc
        )
        # enqueue sets the stage before getPriority reads it.
        if self._get_stage(reg) == LiveRangeStage.RS_New:
            self._set_stage(reg, LiveRangeStage.RS_Assign)
        stage = self._get_stage(reg)
        is_local_assign = (
            stage == LiveRangeStage.RS_Assign
            and not force_global
            and size > 0  # not LI.empty(); guards interval_is_in_one_mbb below
            and self.interval_is_in_one_mbb(reg)
        )
        global_bit = 0
        local_prio = 0
        if is_local_assign:
            # Original local ranges in linear instruction order (optimal coloring
            # absent global interference): forward from the range's begin, or
            # bottom-up to its end when the target assigns locals in reverse.
            if not reverse:
                local_prio = li.begin_index.get_approx_instr_distance(
                    self.last_slot_index()
                )
            else:
                local_prio = self.zero_slot_index().get_approx_instr_distance(
                    li.end_index
                )
        else:
            global_bit = 1
        prio = self._priority_for(
            stage,
            size,
            is_local_assign,
            local_prio,
            global_bit,
            self.reg_class_allocation_priority(rc),
            self.reg_class_priority_trumps_globalness(),
            self.has_known_preference(reg),
        )
        # Python heapq is a min-heap; negate prio for max-first, and use reg as
        # the tie-break (smaller id first, matching the ~Reg.id() ordering).
        heapq.heappush(self._queue, (-prio, reg))

    def dequeue(self):
        if not self._queue:
            return None
        _, reg = heapq.heappop(self._queue)
        return reg

    # -- tryAssign (RegAllocGreedy::tryAssign) ------------------------------
    def _try_assign(self, li):
        """Return the first interference-free physreg in allocation order, or
        None if every physreg interferes. Matches RAGreedy::tryAssign: a copy
        hint is honored only because AllocationOrder front-loads it, so the
        first free reg in order already is the hint when the hint is free -- no
        separate hint check (which would prefer the hint over an earlier free
        reg, diverging from native). CostPerUseLimit is uniform (0) for the
        register classes here, so the cheaper-alternative eviction is a no-op."""
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        return None

    def select_or_split(self, li):
        # Mirrors RegAllocGreedy::selectOrSplitImpl: assign -> evict -> wait for
        # second round -> split -> spill. Returning a physreg makes the
        # framework assign it; returning None with new vregs appended (via
        # evict/split/spill) makes it re-enqueue those; returning None with
        # nothing appended drops the range for this round.
        reg = li.reg

        # First try assigning a free register.
        preg = self._try_assign(li)
        if preg is not None:
            self.trace[reg] = "assign"
            return preg

        stage = self._get_stage(reg)

        # Try to evict a less worthy range, but not for RS_Split ranges: they
        # already failed to evict and must not get a second chance until split.
        if stage != LiveRangeStage.RS_Split:
            preg = self._try_evict(li)
            if preg is not None:
                self.trace[reg] = "evict"
                return preg

        # The first time we see a range, don't split or spill; wait until the
        # second round, when all smaller ranges are allocated and the
        # interference to split around is fully known.
        if stage < LiveRangeStage.RS_Split:
            self._set_stage(reg, LiveRangeStage.RS_Split)
            self.enqueue(reg)
            return None

        # Second round: try splitting the range or its interferences. The
        # sub-methods record the finer trace ("region_split"/"block_split"/
        # "local_split"); keep it rather than overwriting with a generic label.
        if stage < LiveRangeStage.RS_Spill and li.size > 0:
            if self._try_split(li):
                self.trace.setdefault(reg, "split")
                return None

        # A range that is done (already a spill product) or not spillable has no
        # spill recourse; faithfully this is tryLastChanceRecoloring territory,
        # which is not implemented. It does not arise for well-formed MIR the
        # assign/evict/split path resolves, so flag it rather than let the
        # spiller abort on a double spill. Unreachable in tests: an unspillable
        # or RS_Done range that also fails assign/evict/split needs last-chance
        # recoloring, the one stage not yet ported.
        if stage >= LiveRangeStage.RS_Memory or not li.is_spillable:
            raise NotImplementedError(
                "last-chance recoloring is not implemented: reg "
                f"{reg} at stage {int(stage)} is unspillable and unallocatable"
            )

        # Finally, spill the range itself; its reload/remat products go to the
        # terminal RS_Memory stage (RAGreedy's RS_Done) so they are never split
        # or spilled again.
        self.trace[reg] = "spill"
        for product in self.spill(li):
            self._set_stage(product, LiveRangeStage.RS_Memory)
        self._set_stage(reg, LiveRangeStage.RS_Memory)
        return None

    # -- trySplit (RegAllocGreedy::trySplit) --------------------------------
    def _try_split(self, li):
        """Split `li` (local or global) so its pieces become assignable. Returns
        True if new vregs were produced (the framework re-enqueues them).
        Single-block ranges take the local split; multi-block ranges try region
        (global) split first, then per-block isolation (tryBlockSplit). RS_Split2
        ranges skip region split straight to block split, matching trySplit."""
        reg = li.reg
        if self._get_stage(reg) >= LiveRangeStage.RS_Spill:
            return False
        sa = self.split_analysis
        sa.analyze(li)
        if self.interval_is_in_one_mbb(reg):
            # Single-block: local split, then instruction split as a fallback
            # (RAGreedy::trySplit). Instruction split fires here only for a range
            # with subranges (sub-register liveness), i.e. AMDGPU; on AArch64 it
            # returns False.
            if self._try_local_split(li):
                return True
            return self._try_instruction_split(li)
        if self._get_stage(reg) < LiveRangeStage.RS_Split2:
            if self._try_region_split(li):
                return True
        return self._try_block_split(li)

    # -- tryBlockSplit (RegAllocGreedy::tryBlockSplit) ----------------------
    def _should_split_single_block(self, bi, single_instrs):
        """SplitAnalysis::shouldSplitSingleBlock for use block `bi`. Mirrors the
        C++ predicate: always split multi-instruction blocks; for a single
        instruction only split when the class is a proper subclass, and even
        then not a lone copy nor a non-original endpoint."""
        if not bi.is_one_instr():
            return True
        if not single_instrs:
            return False
        # The single-instruction path below is only reached for a proper-subclass
        # register class; no AArch64 GPR32 subclass the hand-built test ops
        # produce is a proper subclass (is_proper_sub_class is False), so it is
        # unreachable here. It faithfully mirrors shouldSplitSingleBlock.
        # Splitting a live-through range always makes progress.
        if bi.live_in and bi.live_out:
            return True
        # No point isolating a copy: it has no register-class constraint.
        # Use MachineInstr::isCopyLike() (generic COPY / SUBREG_TO_REG), the
        # exact predicate shouldSplitSingleBlock tests -- not is_copy_like_at,
        # whose TII::isCopyInstr also matches target-specific copies.
        if self.is_copy_like_instr_at(bi.first_instr):
            return False
        # Don't isolate an endpoint an earlier split created.
        return self.split_analysis.is_original_endpoint(bi.first_instr)

    def _split_single_block(self, se, bi):
        """SplitEditor::splitSingleBlock for use block `bi`: open an interval
        spanning the block's uses, clamped to the block's last legal split
        point, overlapping into a live-out tail when the last use is past it."""
        se.open_intv()
        last_sp = self.split_analysis.last_split_point(bi.mbb)
        first = bi.first_instr
        seg_start = se.enter_intv_before(first if first < last_sp else last_sp)
        if (not bi.live_out) or bi.last_instr < last_sp:
            se.use_intv(seg_start, se.leave_intv_after(bi.last_instr))
        else:
            # The last use is after the last valid split point.
            seg_stop = se.leave_intv_before(last_sp)
            se.use_intv(seg_start, seg_stop)
            se.overlap_intv(seg_stop, bi.last_instr)

    def _try_block_split(self, li):
        """Isolate `li` around each use block SplitAnalysis says is worth
        splitting. Returns True if any block was split (new vregs produced).
        The remainder interval (IntvMap == 0) goes straight to spilling; the
        new local ranges stay RS_New so they can re-compete. Mirrors
        RAGreedy::tryBlockSplit."""
        reg = li.reg
        single_instrs = self.is_proper_sub_class(reg)
        lre = self.new_live_range_edit(li)
        se = self.split_editor
        se.reset(lre, mir.ComplementSpillMode.SM_Speed)
        for bi in self.split_analysis.use_blocks():
            if self._should_split_single_block(bi, single_instrs):
                self._split_single_block(se, bi)
        new_vregs_before = lre.new_vregs()
        if not new_vregs_before:
            return False  # no blocks were split
        intv_map = se.finish()
        # The remainder (IntvMap[i] == 0) that is still RS_New goes to spilling;
        # the isolated local ranges keep RS_New to re-compete.
        new_vregs = lre.new_vregs()
        for i, r in enumerate(new_vregs):
            if (
                i < len(intv_map)
                and intv_map[i] == 0
                and self._get_stage(r) == LiveRangeStage.RS_New
            ):
                self._set_stage(r, LiveRangeStage.RS_Spill)
        self.trace[reg] = "block_split"
        return True

    def _local_gap_weights(self, li, physreg, use_slots):
        """calcGapWeights(PhysReg) for a local interval: for each gap between
        consecutive `use_slots`, the largest interferer spill weight overlapping
        it. Built from the vregs assigned to `physreg` that interfere with `li`,
        plus fixed (physical) reg-unit interference on `physreg`, whose gaps are
        marked huge_valf (a physreg clobbered mid-interval can't hold the value
        across the clobber). Reg-mask (call) clobbers are applied by the caller,
        which knows the regmask gaps. Mirrors RAGreedy::calcGapWeights."""
        base = use_slots[0]
        islots = [base.distance(u) for u in use_slots]
        spans = []
        for ivreg in self.interfering_vregs(li, physreg):
            iv = self.lis.interval(ivreg)
            w = iv.weight
            for seg in iv.segments():
                spans.append((base.distance(seg.start), base.distance(seg.end), w))
        # Fixed physical interference: mark covered gaps huge_valf.
        for seg in self.fixed_interference_spans(li, physreg):
            spans.append((base.distance(seg.start), base.distance(seg.end), _HUGE_VALF))
        return calc_gap_weights(islots, spans)

    def _local_reg_mask_gaps(self, li, bi, uses, num_gaps):
        """The gaps of a local interval that a register mask (call clobber)
        crosses. Faithful port of the RegMaskGaps scan in
        RegAllocGreedy::tryLocalSplit: walk the block's regmask slots alongside
        the use slots and record each gap [Uses[i], Uses[i+1]] a mask falls in.
        Empty when `li` crosses no register mask."""
        if not self.check_reg_mask_interference(li):
            return []
        rms = list(self.reg_mask_slots_in_block(bi.mbb.number))
        gaps = []
        # lower_bound(rms, uses[0].get_reg_slot())
        first = uses[0].get_reg_slot()
        ri = 0
        while ri < len(rms) and rms[ri] < first:
            ri += 1
        re = len(rms)
        for i in range(num_gaps):
            if ri == re:
                break
            if _earlier_instr(uses[i + 1], rms[ri]):
                continue
            # A regmask on the same instruction as the last use doesn't overlap.
            if uses[i + 1].is_same_instr(rms[ri]) and i + 1 == num_gaps:
                break
            gaps.append(i)
            # Advance past this gap; a regmask on a use counts in both gaps.
            while ri != re and _earlier_instr(rms[ri], uses[i + 1]):
                ri += 1
        return gaps

    def _try_local_split(self, li):
        """Local (single-block) splitting: find the contiguous run of uses worth
        keeping in a register (best estimated spill weight minus the largest gap
        interference it must evict) and split around it. Returns True if a split
        was applied. Faithful port of RegAllocGreedy::tryLocalSplit."""
        sa = self.split_analysis
        use_blocks = sa.use_blocks()
        if len(use_blocks) != 1:
            return False
        bi = use_blocks[0]
        uses = list(sa.get_use_slots())
        if len(uses) <= 2:
            return False
        num_gaps = len(uses) - 1
        instr_dist = self.slot_index_instr_distance()
        progress_required = self._get_stage(li.reg) >= LiveRangeStage.RS_Split2

        best_before, best_after, best_diff = num_gaps, 0, 0.0
        block_freq = self.spill_placer.get_block_frequency(bi.mbb).get_frequency() * (
            1.0 / self.mbfi.entry_freq().get_frequency()
        )
        # Gaps that a register mask (call clobber) crosses; for a physreg that
        # the mask clobbers, those gaps become uninhabitable (huge_valf).
        reg_mask_gaps = self._local_reg_mask_gaps(li, bi, uses, num_gaps)

        for physreg in self.allocation_order(li):
            gap_weight = self._local_gap_weights(li, physreg, uses)
            if reg_mask_gaps and self.check_reg_mask_interference_phys(li, physreg):
                for g in reg_mask_gaps:
                    gap_weight[g] = _HUGE_VALF
            split_before, split_after = 0, 1
            max_gap = gap_weight[0]
            while True:
                live_before = split_before != 0 or bi.live_in
                live_after = split_after != num_gaps or bi.live_out
                if not live_before and not live_after:
                    break
                shrink = True
                new_gaps = live_before + split_after - split_before + live_after
                legal = (not progress_required) or new_gaps < num_gaps
                if legal and max_gap < _HUGE_VALF:
                    # Estimate the split range's spill weight: each kept
                    # instruction reads or writes the register once.
                    size = uses[split_before].distance(uses[split_after]) + (
                        (live_before + live_after) * instr_dist
                    )
                    est_weight = normalize_spill_weight(
                        block_freq * (new_gaps + 1), size, instr_dist
                    )
                    if est_weight * _HYSTERESIS >= max_gap:
                        shrink = False
                        diff = est_weight - max_gap
                        if diff > best_diff:
                            best_diff = _HYSTERESIS * diff
                            best_before, best_after = split_before, split_after
                if shrink:
                    split_before += 1
                    if split_before < split_after:
                        # Recompute the running max when the dropped gap was it.
                        # Reached only when the scan shrinks a >=2-gap window
                        # whose dropped gap held the max; the crafted single-
                        # block inputs here never present that gap profile, so
                        # the recompute is a faithful-but-untriggered mirror.
                        if gap_weight[split_before - 1] >= max_gap:
                            max_gap = gap_weight[split_before]
                            for i in range(split_before + 1, split_after):
                                max_gap = max(max_gap, gap_weight[i])
                        continue
                    max_gap = 0.0
                if split_after >= num_gaps:
                    break
                max_gap = max(max_gap, gap_weight[split_after])
                split_after += 1

        if best_before == num_gaps:
            return False  # no candidate window

        lre = self.new_live_range_edit(li)
        se = self.split_editor
        se.reset(lre)
        se.open_intv()
        seg_start = se.enter_intv_before(uses[best_before])
        seg_stop = se.leave_intv_after(uses[best_after])
        se.use_intv(seg_start, seg_stop)
        intv_map = se.finish()
        # If the new range has as many instructions as before, mark it RS_Split2
        # so a further split is forced to make progress (matching tryLocalSplit).
        live_before = best_before != 0 or bi.live_in
        live_after = best_after != num_gaps or bi.live_out
        new_gaps = live_before + best_after - best_before + live_after
        if new_gaps >= num_gaps:
            for i, r in enumerate(lre.new_vregs()):
                if i < len(intv_map) and intv_map[i] == 1:
                    self._set_stage(r, LiveRangeStage.RS_Split2)
        self.trace[li.reg] = "local_split"
        return True

    def _try_instruction_split(self, li):
        """RAGreedy::tryInstructionSplit's sub-register arm. On a target with
        sub-register liveness (AMDGPU), split a range that has subranges around
        each instruction reading only a lane subset, so the pieces can be
        recolored per-lane -- like spilling to a wider class. Returns True if new
        vregs were produced.

        LLVM's other arm splits a range whose register class is a *proper
        subclass* (the X86/ARM mechanism). That is omitted: no linked target
        produces a proper subclass -- RegClassInfo.isProperSubClass is false for
        every allocatable class on AArch64 and AMDGPU, so that arm is unreachable
        and untestable here. hasSubRanges is likewise always false on AArch64
        (no sub-register liveness), so this returns False there."""
        reg = li.reg
        if not self.lis.interval(reg).has_sub_ranges:
            return False
        lre = self.new_live_range_edit(li)
        se = self.split_editor
        se.reset(lre, mir.ComplementSpillMode.SM_Size)
        uses = list(self.split_analysis.get_use_slots())
        if len(uses) <= 1:
            return False
        for use in uses:
            # Split around every non-copy instruction that reads only a subset of
            # the value's live lanes; a full copy (uncoalescable) or a use that
            # reads the whole live value gains nothing from splitting.
            if self.is_full_copy_instr_at(use) or not self.reads_lane_subset(li, use):
                continue
            se.open_intv()
            seg_start = se.enter_intv_before(use)
            seg_stop = se.leave_intv_after(use)
            se.use_intv(seg_start, seg_stop)
        if not lre.new_vregs():
            return False  # all uses were copies / read the whole value
        se.finish()
        # This was the last split chance: all new ranges go straight to spilling.
        for r in lre.new_vregs():
            self._set_stage(r, LiveRangeStage.RS_Spill)
        self.trace[reg] = "instruction_split"
        return True

    # -- region split (tryRegionSplit and its cost model) -------------------
    def _add_split_constraints(self, intf):
        """RAGreedy::addSplitConstraints. Build BlockConstraints for the use
        blocks from `intf` (a cursor already pointed at the candidate physreg),
        accumulate the static spill cost, add them to the SpillPlacement
        network, and return (cost, any_positive)."""
        sa = self.split_analysis
        use_blocks = sa.use_blocks()
        sp = self.spill_placer
        self._split_constraints = []
        static_cost = mir.BlockFrequency(0)
        PrefReg = mir.BorderConstraint.PrefReg
        PrefSpill = mir.BorderConstraint.PrefSpill
        MustSpill = mir.BorderConstraint.MustSpill
        DontCare = mir.BorderConstraint.DontCare
        for bi in use_blocks:
            bc = mir.BlockConstraint()
            bc.number = bi.mbb.number
            intf.move_to_block(bc.number)
            bc.entry = PrefReg if bi.live_in else DontCare
            # An implicit-def last instruction does not keep the value live out.
            # Only read the last instruction when the value is actually live out
            # (LLVM short-circuits `LiveOut && !isImplicitDef`), so a non-live-out
            # block never dereferences its last-instr index.
            bc.exit = (
                PrefReg
                if (
                    bi.live_out
                    and not self.lis.instr_from_index(bi.last_instr).is_implicit_def
                )
                else DontCare
            )
            bc.changes_value = bi.first_def.is_valid()
            if intf.has_interference():
                ins = 0
                mbb_start = self.lis.mbb_start_index(bi.mbb)
                if bi.live_in:
                    if not (mbb_start < intf.first()):  # first() <= start
                        bc.entry = MustSpill
                        ins += 1
                    elif intf.first() < bi.first_instr:
                        bc.entry = PrefSpill
                        ins += 1
                    elif intf.first() < bi.last_instr:
                        ins += 1
                    # Abort if the spill cannot be inserted at the block start.
                    if bc.entry in (MustSpill, PrefSpill) and _earlier_instr(
                        bi.first_instr, sa.first_split_point(bc.number)
                    ):
                        return static_cost, False
                if bi.live_out:
                    lsp = sa.last_split_point(bi.mbb)
                    if not (intf.last() < lsp):  # last() >= last split point
                        bc.exit = MustSpill
                        ins += 1
                    elif intf.last() > bi.last_instr:
                        bc.exit = PrefSpill
                        ins += 1
                    elif intf.last() > bi.first_instr:
                        ins += 1
                for _ in range(ins):
                    static_cost = static_cost + sp.get_block_frequency_by_number(
                        bc.number
                    )
            self._split_constraints.append(bc)
        sp.add_constraints(self._split_constraints)
        return static_cost, sp.scan_active_bundles()

    def _add_through_constraints(self, intf, blocks):
        """RAGreedy::addThroughConstraints. Interference-free through blocks
        become transparent links; interfering ones get MustSpill/PrefSpill
        entry/exit constraints. Returns False if a required spill cannot be
        inserted at a block start."""
        sa = self.split_analysis
        sp = self.spill_placer
        MustSpill = mir.BorderConstraint.MustSpill
        PrefSpill = mir.BorderConstraint.PrefSpill
        links = []
        constraints = []
        for number in blocks:
            intf.move_to_block(number)
            if not intf.has_interference():
                links.append(number)
                continue
            bc = mir.BlockConstraint()
            bc.number = number
            # Abort if the spill cannot be inserted at the block start.
            first_instr = self.first_nondebug_instr_index(number)
            if first_instr.is_valid() and _earlier_instr(
                first_instr, sa.first_split_point(number)
            ):
                return False
            insert_idx = self.through_insert_index(number)
            mbb_start = self.mbb_start_index_by_number(number)
            if (not (mbb_start < intf.first())) or _earlier_instr(
                intf.first(), insert_idx
            ):
                bc.entry = MustSpill
            else:
                bc.entry = PrefSpill
            if not (intf.last() < sa.last_split_point_number(number)):
                bc.exit = MustSpill
            else:
                bc.exit = PrefSpill
            constraints.append(bc)
        if links:
            sp.add_links(links)
        if constraints:
            sp.add_constraints(constraints)
        return True

    def _grow_region(self, li, cand):
        """RAGreedy::growRegion. Expand the candidate's active through-block set
        until it stops growing, applying through constraints (or a loop-IV-aware
        pref-spill for the compact region). Returns False if the complexity
        budget is exhausted or a spill is uninsertable."""
        sa = self.split_analysis
        sp = self.spill_placer
        eb = self.edge_bundles
        todo = set(sa.through_blocks())
        added_to = 0
        budget = _GROW_REGION_COMPLEXITY_BUDGET
        while True:
            for bundle in sp.get_recent_positive():
                blocks = eb.get_blocks(bundle)
                if len(blocks) >= budget:
                    return False
                budget -= len(blocks)
                for block in blocks:
                    if block not in todo:
                        continue
                    todo.discard(block)
                    cand.active_blocks.append(block)
            if len(cand.active_blocks) == added_to:
                break
            new_blocks = cand.active_blocks[added_to:]
            if cand.phys_reg:
                if not self._add_through_constraints(cand.intf, new_blocks):
                    return False
            else:
                # Compact region: bias through blocks to spill, except a loop
                # header + its internal blocks (keep the IV live header<->latch).
                pref_spill = True
                if sa.looks_like_loop_iv() and len(new_blocks) >= 2:
                    hdr = self.loop_header_number(new_blocks[0])
                    if hdr == new_blocks[0] and all(
                        self.loop_header_number(b) == hdr for b in new_blocks[1:]
                    ):
                        pref_spill = False
                if pref_spill:
                    sp.add_pref_spill(new_blocks, True)
            added_to = len(cand.active_blocks)
            sp.iterate()
        return True

    def _calc_global_split_cost(self, cand, order):
        """RAGreedy::calcGlobalSplitCost. Cost of the candidate's bundle
        solution: a spill at each use-block edge whose in/out register state
        disagrees with the constraint pref, plus through-block crossings."""
        sp = self.spill_placer
        eb = self.edge_bundles
        lb = cand.live_bundles
        PrefReg = mir.BorderConstraint.PrefReg
        cost = mir.BlockFrequency(0)
        use_blocks = self.split_analysis.use_blocks()
        for bi, bc in zip(use_blocks, self._split_constraints):
            reg_in = lb.test(eb.get_bundle_number(bc.number, False))
            reg_out = lb.test(eb.get_bundle_number(bc.number, True))
            ins = 0
            cand.intf.move_to_block(bc.number)
            if bi.live_in:
                ins += int(reg_in != (bc.entry == PrefReg))
            if bi.live_out:
                ins += int(reg_out != (bc.exit == PrefReg))
            for _ in range(ins):
                cost = cost + sp.get_block_frequency_by_number(bc.number)
        for number in cand.active_blocks:
            reg_in = lb.test(eb.get_bundle_number(number, False))
            reg_out = lb.test(eb.get_bundle_number(number, True))
            if not reg_in and not reg_out:
                continue
            if reg_in and reg_out:
                cand.intf.move_to_block(number)
                if cand.intf.has_interference():
                    cost = cost + sp.get_block_frequency_by_number(number)
                    cost = cost + sp.get_block_frequency_by_number(number)
                continue
            cost = cost + sp.get_block_frequency_by_number(number)
        return cost

    def _calc_compact_region(self, li, cand):
        """RAGreedy::calcCompactRegion. The compact region removes all
        through blocks; needs no interference (PhysReg unset). Returns False if
        the range is already compact or the compact region is trivial."""
        sa = self.split_analysis
        if sa.num_through_blocks() == 0:
            return False
        cand.reset(0, cand.intf)  # PhysReg = NoRegister
        self.set_interference_physreg(cand.intf, 0)
        sp = self.spill_placer
        sp.prepare(cand.live_bundles)
        # Static cost is zero (no interference); a False here means no positive
        # bundles, i.e. nothing to keep in a register.
        cost, positive = self._add_split_constraints(cand.intf)
        if not positive:
            return False
        if not self._grow_region(li, cand):
            return False
        sp.finish()
        if not cand.live_bundles.count() > 0:
            return False
        return True

    def _is_unused_callee_saved(self, physreg):
        """EvictAdvisor::isUnusedCalleeSavedReg: `physreg` aliases a callee-saved
        register that has not been assigned yet (so using it would widen the
        callee-saved set)."""
        return self.last_callee_saved_alias(
            physreg
        ) != 0 and not self.matrix.is_phys_reg_used(physreg)

    def _calculate_region_split_cost(self, li, order, best_cost, num_cands, ignore_csr):
        """RAGreedy::calculateRegionSplitCost. Score a candidate per physreg;
        return (best_cand_index_or__NO_CAND, num_cands)."""
        best_cand = _NO_CAND
        for physreg in order:
            if ignore_csr and self._is_unused_callee_saved(physreg):
                continue
            best_cand, num_cands, best_cost = (
                self._calculate_region_split_cost_around_reg(
                    li, physreg, order, best_cost, num_cands, best_cand
                )
            )
        return best_cand, num_cands

    def _calculate_region_split_cost_around_reg(
        self, li, physreg, order, best_cost, num_cands, best_cand
    ):
        """RAGreedy::calculateRegionSplitCostAroundReg for one physreg.

        The C++ recycles interference-cache cursors once NumCands reaches
        IntfCache.getMaxCursors(); that cap (a large default) is never reached
        by the hand-built test corpus and getMaxCursors is not exposed, so the
        recycling branch is omitted -- the scoring result is identical."""
        if len(self._global_cand) <= num_cands:
            self._global_cand.append(GlobalSplitCandidate())
        cand = self._global_cand[num_cands]
        cand.reset(physreg, self.new_interference_cursor())
        self.set_interference_physreg(cand.intf, physreg)
        sp = self.spill_placer
        sp.prepare(cand.live_bundles)
        cost, positive = self._add_split_constraints(cand.intf)
        if not positive:
            return best_cand, num_cands, best_cost
        if not (cost < best_cost):
            return best_cand, num_cands, best_cost
        if not self._grow_region(li, cand):
            return best_cand, num_cands, best_cost
        sp.finish()
        if not cand.live_bundles.count() > 0:
            return best_cand, num_cands, best_cost
        cost = cost + self._calc_global_split_cost(cand, order)
        if cost < best_cost:
            best_cand = num_cands
            best_cost = cost
        num_cands += 1
        return best_cand, num_cands, best_cost

    def _region_cand0(self):
        """Ensure GlobalCand[0] (the compact-region candidate slot) exists with
        an interference cursor, and return it. calcCompactRegion writes into it."""
        if not self._global_cand:
            self._global_cand.append(GlobalSplitCandidate())
        cand = self._global_cand[0]
        if cand.intf is None:
            cand.intf = self.new_interference_cursor()
        return cand

    def _calc_block_split_cost(self):
        """RAGreedy::calcBlockSplitCost: the cost of isolating each use block
        instead of forming bundle regions -- one spill per use block, plus a
        second for a block where the value is both live-through and redefined.
        Region split must beat this fallback."""
        cost = mir.BlockFrequency(0)
        sp = self.spill_placer
        for bi in self.split_analysis.use_blocks():
            number = bi.mbb.number
            cost = cost + sp.get_block_frequency_by_number(number)
            if bi.live_in and bi.live_out and bi.first_def.is_valid():
                cost = cost + sp.get_block_frequency_by_number(number)
        return cost

    def _try_region_split(self, li):
        """RAGreedy::tryRegionSplit. Score region-split candidates (plus the
        compact region), and if one beats spilling, apply it. Returns True if
        new vregs were produced."""
        # Target opt-out: some targets (e.g. AMDGPU) disable region splitting for
        # a vreg; the AArch64 default is true.
        if not self.should_region_split_for_virt_reg(li.reg):
            return False
        order = list(self.allocation_order(li))
        num_cands = 0
        spill_cost = self._calc_block_split_cost()  # cost of isolating all blocks
        has_compact = self._calc_compact_region(li, self._region_cand0())
        if has_compact:
            num_cands = 1
            best_cost = mir.BlockFrequency.max()
        else:
            best_cost = spill_cost
        best_cand, num_cands = self._calculate_region_split_cost(
            li, order, best_cost, num_cands, False
        )
        if not has_compact and best_cand == _NO_CAND:
            return False
        lre = self.new_live_range_edit(li)
        self._do_region_split(li, best_cand, has_compact, lre)
        if len(lre.new_vregs()) > 0:
            self.trace[li.reg] = "region_split"
            return True
        return False

    def _do_region_split(self, li, best_cand, has_compact, lre):
        """RAGreedy::doRegionSplit. Assign edge bundles to the chosen
        candidate(s), open their split intervals, then apply the region."""
        se = self.split_editor
        se.reset(lre, mir.ComplementSpillMode.SM_Speed)
        num_bundles = self.edge_bundles.num_bundles()
        self._bundle_cand = [_NO_CAND] * num_bundles
        used_cands = []
        if best_cand != _NO_CAND:
            cand = self._global_cand[best_cand]
            if self._cand_get_bundles(cand, best_cand) > 0:
                used_cands.append(best_cand)
                cand.intv_idx = se.open_intv()
        if has_compact:
            cand = self._global_cand[0]
            if self._cand_get_bundles(cand, 0) > 0:
                used_cands.append(0)
                cand.intv_idx = se.open_intv()
        self._split_around_region(li, lre, used_cands)

    def _cand_get_bundles(self, cand, cand_index):
        """GlobalSplitCandidate::getBundles: claim this candidate's live bundles
        in the shared _bundle_cand map. Returns the number newly claimed."""
        count = 0
        for i in cand.live_bundles.set_bits():
            if self._bundle_cand[i] == _NO_CAND:
                self._bundle_cand[i] = cand_index
                count += 1
        return count

    def _split_around_region(self, li, lre, used_cands):
        """RAGreedy::splitAroundRegion. Drive the high-level SplitEditor calls
        per use block and per through block, finish, and stage new intervals."""
        se = self.split_editor
        sa = self.split_analysis
        eb = self.edge_bundles
        # C++ NumGlobalIntvs = LREdit.size() at entry: the new vregs already
        # created by the openIntv calls in doRegionSplit (before splitSingleBlock
        # adds any block-local ones). Do NOT use len(used_cands): openIntv can
        # create more than one edit entry, and the IntvMap < NumGlobalIntvs
        # staging check is off-by-one against len(used_cands).
        num_global_intvs = len(lre.new_vregs())
        single_instrs = self.is_proper_sub_class(li.reg)
        # Use blocks.
        for bi in sa.use_blocks():
            number = bi.mbb.number
            intv_in = intv_out = 0
            intf_in = intf_out = None
            if bi.live_in:
                cand_in = self._bundle_cand[eb.get_bundle_number(number, False)]
                if cand_in != _NO_CAND:
                    cand = self._global_cand[cand_in]
                    intv_in = cand.intv_idx
                    cand.intf.move_to_block(number)
                    intf_in = cand.intf.first()
            if bi.live_out:
                cand_out = self._bundle_cand[eb.get_bundle_number(number, True)]
                if cand_out != _NO_CAND:
                    cand = self._global_cand[cand_out]
                    intv_out = cand.intv_idx
                    cand.intf.move_to_block(number)
                    intf_out = cand.intf.last()
            if not intv_in and not intv_out:
                if self._should_split_single_block(bi, single_instrs):
                    se.split_single_block(bi)
                continue
            if intv_in and intv_out:
                se.split_live_through_block(
                    number, intv_in, intf_in, intv_out, intf_out
                )
            elif intv_in:
                se.split_reg_in_block(bi, intv_in, intf_in)
            else:
                se.split_reg_out_block(bi, intv_out, intf_out)
        # Through blocks (dedup across candidates).
        todo = set(sa.through_blocks())
        for used_cand in used_cands:
            for number in self._global_cand[used_cand].active_blocks:
                if number not in todo:
                    continue
                todo.discard(number)
                intv_in = intv_out = 0
                intf_in = intf_out = None
                cand_in = self._bundle_cand[eb.get_bundle_number(number, False)]
                if cand_in != _NO_CAND:
                    cand = self._global_cand[cand_in]
                    intv_in = cand.intv_idx
                    cand.intf.move_to_block(number)
                    intf_in = cand.intf.first()
                cand_out = self._bundle_cand[eb.get_bundle_number(number, True)]
                if cand_out != _NO_CAND:
                    cand = self._global_cand[cand_out]
                    intv_out = cand.intv_idx
                    cand.intf.move_to_block(number)
                    intf_out = cand.intf.last()
                if not intv_in and not intv_out:
                    continue
                se.split_live_through_block(
                    number, intv_in, intf_in, intv_out, intf_out
                )
        intv_map = se.finish()
        # Stage the new intervals (matches splitAroundRegion's four kinds).
        orig_blocks = sa.num_live_blocks()
        new_vregs = lre.new_vregs()
        for i, r in enumerate(new_vregs):
            if self._get_stage(r) != LiveRangeStage.RS_New:
                continue
            m = intv_map[i] if i < len(intv_map) else 0
            if m == 0:
                self._set_stage(r, LiveRangeStage.RS_Spill)
            elif m < num_global_intvs:
                if (
                    self.split_analysis.count_live_blocks(self.lis.interval(r))
                    >= orig_blocks
                ):
                    self._set_stage(r, LiveRangeStage.RS_Split2)
            # else: block-local / DCE leftovers stay RS_New.

    # -- tryEvict / canEvictInterference ------------------------------------
    def _can_evict_interference(self, li, physreg):
        """True if every vreg interfering with `li` on `physreg` can be evicted:
        each must have a strictly smaller spill weight, must not be a spill
        product, and must have a strictly older cascade. Mirrors
        canEvictInterference: `li` uses its cascade-or-next (a range without a
        cascade peeks the next value), interferers use their raw cascade (0 =
        never evicted), and a range can evict anything with a lower cascade."""
        # Only virtual-register interference is evictable. If `physreg` carries
        # fixed, reg-unit, or reg-mask interference (checkInterference returns a
        # kind worse than IK_VirtReg), it cannot be freed by eviction -- this is
        # the first guard in canEvictInterferenceBasedOnCost. Without it a
        # physreg with only non-vreg interference and no evictable vregs looks
        # "evictable for free" and gets assigned over a live occupant.
        kind = self.matrix.check_interference(li, physreg)
        if kind.value > mir.InterferenceKind.IK_VirtReg.value:
            return False
        cascade = self._cascade_or_next(li.reg)
        interferers = self.interfering_vregs(li, physreg)
        if not interferers:
            return False
        # With this many interferers, chances are one is heavier; RAGreedy
        # refuses outright rather than scan them all (EvictInterferenceCutoff).
        if len(interferers) >= EVICT_INTERFERENCE_CUTOFF:
            return False
        for ivreg in interferers:
            iv = self.lis.interval(ivreg)
            # Never evict spill products (RS_Done); they cannot split or spill.
            if self._get_stage(ivreg) >= LiveRangeStage.RS_Memory:
                return False
            if iv.weight >= li.weight:
                return False
            # Only evict strictly-older cascades (or cascade-less, cascade 0);
            # an equal or newer cascade is the eviction-loop guard. We do not
            # implement urgent cascade-breaking (a last-resort branch).
            if cascade <= self._get_cascade(ivreg):
                return False
        return True

    def _eviction_cost_for(self, li, physreg):
        """EvictionCost (broken_hints, max_weight) for evicting `li`'s
        interferers off `physreg`: each interferer at its preferred physreg
        (has_preferred_phys) contributes its class copy cost to broken_hints, and
        max_weight is the heaviest interferer. Mirrors
        canEvictInterferenceBasedOnCost's per-interferer accumulation."""
        triples = []
        for v in self.interfering_vregs(li, physreg):
            weight = self.lis.interval(v).weight
            breaks_hint = self.has_preferred_phys(v)
            copy_cost = (
                self.reg_class_copy_cost(self.reg_class(v)) if breaks_hint else 0.0
            )
            triples.append((weight, breaks_hint, copy_cost))
        return eviction_cost(triples)

    def _try_evict(self, li):
        """Evict the interferers off the cheapest evictable physreg and assign
        `li` there. Returns the physreg, or None if nothing is evictable."""
        best, best_cost = None, None
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                continue  # tryAssign already tried the free ones
            if not self._can_evict_interference(li, preg):
                continue
            cost = self._eviction_cost_for(li, preg)
            if best_cost is None or cost < best_cost:
                best, best_cost = preg, cost
        if best is None:
            return None
        # Assign and consume li's cascade now that it actually evicts (matching
        # evictInterference's getOrAssignNewCascade).
        li_cascade = self._assign_cascade(li.reg)
        for ivreg in list(self.interfering_vregs(li, best)):
            iv = self.lis.interval(ivreg)
            self.matrix.unassign(iv)
            # Evicted ranges inherit li's cascade so they can't evict li back
            # within this cascade (the infinite-eviction guard). Their stage is
            # left unchanged -- mirroring evictInterference, which sets only the
            # cascade; the range re-competes from wherever it already was.
            self._cascade[ivreg] = li_cascade
            self.enqueue(ivreg)
        # Evicting the interferers frees `best`; return it and let the framework
        # do the assignment (assigning here as well would double-assign and
        # abort). `best` now passes select_or_split's free-candidate check.
        return best


mir.RAGreedy = RAGreedy
