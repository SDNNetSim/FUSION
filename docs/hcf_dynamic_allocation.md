# Dynamic HCF allocation in SDNController

`SDNController.handle_hcf_event()` is the opt-in, full-HCF WDM allocator.
The existing `handle_event()`, `_process_single_path()`, and legacy finalizer
remain unchanged. `SpectrumAssignment.get_spectrum_hcf()` owns the wavelength
loop, candidate GSNR calls, power-budget checks, probe generation, opportunity
scoring, and selection on one path. Its `rank_hcf_first_channels()` method also
owns comparison across paths for the `first_channel_ranked` policy.
`SDNController` owns the outer path loop,
power ledger, reservations, commit, and departure. `fusion/core/hcf.py` contains immutable candidate/mode/result
records and power-unit conversion only. There is no second GSNR model or KSP
implementation.

The request flow reuses `_setup_routing()` / `Routing.get_route()`, then calls
`SpectrumAssignment.get_spectrum_hcf()` for each path. That method reuses
`_calculate_slots_needed()`, `find_common_channels_on_paths()`, and the object's
existing `SnrMeasurements.check_gsnr_hcf()` / `calculate_hcf_snr_link()`.
Probe routes use the existing `KShortestPath.get_paths()` implementation.

```text
SDNController.handle_hcf_event()
  -> existing routing / outer path loop
  -> SpectrumAssignment.get_spectrum_hcf(path, ...)
       -> existing common-channel search
       -> existing HCF GSNR calculation
       -> internal power-budget check against the controller's ledger
       -> internal power/opportunity scoring using virtual reservations
       -> HCFSpectrumResult(candidate, score, diagnostics)
  -> SDNController commits spectrum and power together
```

Spectrum search is read-only: it returns an immutable result without reserving
resources or publishing candidate values to `spectrum_props`. The controller
provides a read-only view of its power ledger and an adapter to existing KSP
for probe routes, then publishes only the selected candidate during commitment.
It supplies request-eligible modes for the actual search and the full mode
catalog for probes, which can have different requested bitrates. No policy
callbacks are needed. Existing `get_spectrum()` remains unchanged.
`get_hcf_free_channels()` and `get_hcf_candidates()` are also reused by the
opportunity probes. Only real-request searches materialize candidate objects
and amplifier-demand lists. Probe scoring uses compact transmission records
(mode index, forward power, reverse power, minimum GSNR) and two live residual
power minima per path/band. Under the existing common-power-per-direction model,
these minima are equivalent to testing every amplifier demand individually.
The transmission records have a separate LRU cache capped at 4,096 entries
and 16 MiB of accounted storage. Both directional physical signatures, the mode
catalog, power objective/grid, margin and transceiver ceiling are in its key.
Empty (physically infeasible) records are also reusable. Spectrum availability,
per-amplifier budgets, virtual additions, headroom and scores are never cached.
Iteration reset clears transmission records as well as the other caches.
Numeric span/path noise coefficients and fixed/discrete GSNR results
are reused in SnrMeasurements. Each calculator retains at most 16,384 entries
and 16 MiB of accounted key/value storage, evicting least-recently-used entries.
This is a cache-storage bound, not a whole-process memory limit. Keys include
resolved physical inputs; changed noise parameters or power cannot reuse an old
result. Threshold, bitrate and margin checks still run on every GSNR query.
The existing iteration reset clears this physical cache. Together, physical
and transmission caches are capped at 32 MiB of accounted storage per allocator,
plus the separately retained small configured probe set.
Probe requests, normalized weights and K-shortest paths are prepared once and
reused within an iteration. Only one probe set is retained, with size determined
by the configured probe count and K paths; it does not grow with arrivals.
The controller clears it at iteration reset. If probe configuration or topology
is changed manually, call clear_hcf_probe_cache() before another search.
Spectrum availability, amplifier budgets and opportunity scores are always
evaluated against current resources; none of these results are cached.
`hcf_power_fits()` is also reused by the controller to validate available power
immediately before committing. Only the controller changes the power ledger.

## Enable

Use `fusion/configs/templates/hcf_dynamic_power.ini` as a complete example
configuration with the normal FUSION launcher. The existing default config
has not been changed. The important switch is in `[general_settings]`:

```ini
resource_allocation_mode = hcf_power_opportunity
hcf_selection_policy = joint
```

`resource_allocation_mode` defaults to `legacy` when absent. The new allocator
requires the legacy engine: an enabled `FUSION_USE_ORCHESTRATOR` override is
rejected. Set that environment variable to `0` if it was enabled.

Supported initial scope: full HCF, fixed grid, one core, one single-carrier
channel per accepted request, bidirectional service, zero additional guard
slots, `max_segments=1`, no slicing/grooming/partial service/protection/failure
recovery or transponder limits, and `snr_recheck=False`. The mode's symbol rate
and roll-off must fit within the fixed channel spacing. A request that cannot
fit any one carrier is blocked; it is never split.

The loader keeps optional topology settings nested, so `bi_directional=True`
belongs in `[topology_settings]`. Put `k_paths` in `[routing_settings]`.

## Request flow

The simulation selects the new method at both existing arrival and release
call sites. As with the original method, it first populates `sdn_props` from
the request. Direct callers must do the same.

```python
controller.handle_hcf_event(request, request_type="arrival")
# Departure: restore the saved path and lightpath_id_list, as SimulationEngine does.
controller.handle_hcf_event(request, request_type="release")
```

For arrivals, the controller visits the existing routing module's K-shortest
paths (length by default, respecting its configured routing weight).
It ignores the legacy SMF-style modulation reach filter and checks explicit
HCF transmission modes instead. Within each path it scans common free
wavelengths, evaluates compatible mode/power pairs, and compares feasible
allocations. The original policies accept the best on the first feasible path;
`first_channel_ranked` instead compares one channel from every path. Forced paths,
channel index, band, core 0, and modulation names are supported.

GSNR evaluation concerns only candidate channels. Established channels are
never retuned or re-evaluated. Candidate threshold feasibility, occupancy and remaining power are checked on each arrival; numeric physical results may be reused.

Each wavelength's lowest-power QoT-feasible mode/power pair dominates its
higher-power feasible pairs for this resource-only objective. The selector
scores that pair once, after testing the complete power grid for physical
feasibility. It does **not** assume GSNR increases monotonically with power.
Ties use lower power, configured band order, channel index, smaller adequate
carrier capacity, then mode name.

## Configuration

All `hcf_*` keys are optional schema entries in `[general_settings]`, but
the new mode requires an explicit power grid and per-band amplifier budgets.
By default it uses `SNRProps.bandwidth_mapping_dict`, `bits_per_symbol`, and
`req_snr`, exactly as the existing HCF calculator does. `hcf_modes` is an
optional override for calibrated transmission modes. JSON-valued fields can
also be passed as native lists/dictionaries in `engine_props`.

```ini
hcf_modes = [{"name":"QPSK","net_rate_gbps":200,"baud_rate_gbaud":50,"threshold_db":6.72,"roll_off":0.1}]
hcf_power_levels_dbm = [-3, 0, 3]
hcf_amplifier_budget_dbm = {"c": 20}
hcf_margin_db = 1
hcf_transceiver_snr_db = 25
hcf_power_weight = 0.5
hcf_probe_count = 8
hcf_probe_seed = 7
```

These numbers are illustrative, not a validated line-system/transceiver
configuration. Net bitrates, actual symbol rates, FEC thresholds, amplifier
headroom, noise figures, and spectral loss profiles must be calibrated for
the paper. Mode rates are explicit: the allocator does not infer FEC or
symbol rate from bits per symbol when an explicit mode override is supplied.
Without an override, the existing simulator's conversion is retained.
In this HCF allocator, `optimized_power` is overridden by explicit
candidate powers selected from `hcf_power_levels_dbm`.

Policies:

- `first_feasible`: accept the first wavelength having a feasible mode/power.
  Use a one-element power grid for the fixed-power baseline, or multiple
  powers for the adaptive-power baseline.
- `power`: score every feasible wavelength on the current path using the
  average fraction of residual amplifier headroom consumed.
- `joint`: `w * power_scarcity + (1-w) * opportunity_loss`, where
  `w = hcf_power_weight` and opportunity loss is normalized by the current
  opportunity value. The loss is zero if that value is zero.
- `first_channel_ranked`: use the same joint score, but inspect only the first
  common-free channel on each candidate path, then rank those path candidates
  together. Band order follows `band_list`; within the first band having free
  channels, the lowest channel index is used. This is one channel per path,
  not one per band. Forced band/channel constraints take precedence.

### First-channel ranked policy

To enable this alternative, retain `resource_allocation_mode = hcf_power_opportunity`
and set in `[general_settings]`:

```ini
hcf_selection_policy = first_channel_ranked
hcf_power_weight = 0.5
```

For each arrival:

1. Obtain candidate routes from the existing routing module.
2. Select each path's first channel free on all links in both directions.
3. Evaluate compatible mode/power pairs on that channel using the existing HCF
   GSNR routine. Keep its lowest-power feasible pair and check amplifier budgets.
   If that channel fails, discard the path; do not advance to another channel.
4. Compute the current probe opportunity once. Score each surviving candidate
   using its virtual reservation and the joint power/opportunity objective.
5. Sort by ascending score, then launch power, then original route index.
6. Visit candidates in rank order and commit the first whose spectrum and power
   are still available. Normally rank 1 succeeds in the sequential event loop.
   Unexpected commit errors still propagate with transaction rollback.

The implementation computes the shared opportunity baseline before the path
searches; it is the same unchanged network state. With power weight one,
opportunity calculations are skipped. Probe feasibility still uses all probe
paths/channels and the complete transmission-mode catalog; only the current
request's candidate set is restricted. Established lightpaths are not re-evaluated.

With K candidate paths, this policy performs at most K virtual opportunity
evaluations plus one baseline, instead of scoring every feasible channel on
visited paths. This bounds candidate-scoring work, but probe enumeration and
physical calculations can still be costly. No runtime improvement
factor or blocking improvement is assumed without measurement.

This restriction can block a request even when a later channel would pass GSNR.
It also changes route selection from the original sequential path preference.
Compare runtime per arrival, request/bandwidth blocking, and carried traffic
against `joint`, `power`, and `first_feasible` using identical traffic seeds,
physical parameters, power grid, and probe set. The active config is not changed
automatically, and a running simulation keeps the code/config it already loaded.

`hcf_last_result` includes `ranked_candidates` and, on acceptance, `selected_rank`.
The existing `free_channels` counter records inspected channels (at most K),
not every unoccupied channel in the network. Ranking is read-only; commitment
and departure reuse the existing HCF resource lifecycle.

For probes q, opportunity is `sum(weight_q * log(1 + N_q))`. `N_q` counts
unique feasible `(path, band, wavelength)` placements, once regardless of
how many modes/powers can serve that placement. Score evaluation uses a
virtual reservation, without changing actual state. This count is a heuristic
and does not represent simultaneously supportable traffic.

Probes are either a reproducible sample of uniform source/destination pairs
and the configured bitrate distribution, or an explicit JSON list:

```ini
hcf_probes = [{"source":"1","destination":"5","bandwidth":200,"weight":0.6},{"source":"2","destination":"8","bandwidth":100,"weight":0.4}]
```

Use explicit probes for a nonuniform spatial traffic matrix. Node IDs must
match topology IDs. Weights are normalized. Sampling uses its own seeded RNG
and does not consume traffic RNG state or inspect the future request list.
Exact scoring of the probe set is implemented first; use modest probe counts
and measure runtime before scaling experiments.

## Physical model and budgets

All ASE, nonlinear-interference and intermodal-interference equations remain
in the existing `SnrMeasurements.calculate_hcf_snr_link()` implementation.
`check_gsnr_hcf()` now accepts optional explicit candidate inputs (path, band,
channel, mode, power, bitrate, symbol rate, threshold/margin and transceiver
SNR). Existing no-argument calls still obtain these from spectrum/SNR state.
The SDN adapter calls this routine for each direction and requires both to
pass; it does not contain noise equations or temporarily modify global power.

Integration corrections in the existing routines:

- Read dispersion from `dispersion` and gamma from `non_linearity`, matching
  their declared units in the topology generator.
- Read the band frequency origin from the fiber dictionary as the lower band
  edge and use channel centers `f_start + (channel + 0.5) * spacing`, checking
  a consistent grid along the path.
- Sum each link/span's inverse GSNR, including the last partial span, instead
  of applying the final link's parameters to the entire route.
- Allow an explicit candidate launch power to override internal optimization.
- Honor the amplifier gain limit, optional margin, and optional transceiver
  ceiling (once per path direction).

The existing formula uses a fixed fully populated WDM comb: its channel count
comes from configured spectrum capacity, not current occupancy. That modeling
assumption is retained. It is an approximation for mixed channel powers, and
should be validated for the experiment; this refactor does not introduce a
new physical model or establish accuracy for all loading conditions. Established
channels are not re-evaluated when requests arrive or depart.

Optional `fiber['hcf']['band_profiles'][band]` can override physical scalars
with calibrated values or per-channel lists. The new SDN algorithm's budgets
remain separate from the physical calculator. `max_amplifier_gain` is a gain
limit, not an output-power budget. Budget keys identify
`(source, destination, span_index, band)`; opposite directions have separate
amplifiers even though their spectrum matrices are aliased. Each accepted
bidirectional lightpath reserves its launch power in both directions. A scalar
`power_budget_dbm` in the HCF link/band profile overrides the configured
per-band budget. These are usable signal-power budgets after required ASE and
headroom derating. Separate bands have separate budgets in this implementation.

The former standalone `span_inverse_gsnr()` and calibrated-SCI approximation
have been removed. `sci_eta_w_inv2` is not an input to the reused HCF model.

## State and diagnostics

Commit records exactly one lightpath ID and stores its exact power demands.
A failed allocation rolls back touched spectrum, usage counters, and request
bookkeeping. Departure frees the channel and those stored power amounts.
Iteration reset clears active HCF metadata and power reservations
before IDs are reused. Treat physical configuration as immutable during an
iteration; restart/reset before changing model inputs.

`hcf_last_result` exposes the latest arrival diagnostics. HCF runs persist
`hcf_decisions_list` through the normal statistics output, including admission,
blocking reason, score, route, channel, mode, launch power, GSNR/margin,
requested/carrier bitrates, and total allocation time. Carrier capacity can
exceed requested bitrate; utilization uses requested/carrier capacity rather
than falsely reporting every channel as fully utilized.

Blocking classification: no route/eligible mode Ã¢â€ â€™ `distance`; no common free
channel Ã¢â€ â€™ `congestion`; no candidate passing GSNR Ã¢â€ â€™ `xt_threshold`; physically
feasible candidates but insufficient amplifier headroom Ã¢â€ â€™ `hcf_power`.
Per-arrival counters retain the distinction between free and QoT-feasible
channels. `hcf_power_used` and `hcf_allocations` expose active power state.

## Tests

```powershell
.venv/Scripts/python.exe -m pytest fusion/core/tests/test_hcf_controller.py -q
```

Tests cover path/wavelength order, actual opportunity selection, power limits,
QoT reference calculations and partial spans, reverse-direction feasibility,
rollback, release/reset, deterministic probes, no established-channel rechecks,
and a real engine/configuration/statistics/persistence smoke test.
