# Dynamic HCF power and wavelength comparisons

These policies use fixed-grid single-carrier WDM, existing K-shortest paths,
existing channel continuity checks and the existing HCF noise model.
The active `fusion/configs/config.ini` is not modified.

## Four standalone experiment configs

| File in fusion/configs/experiments | Power objective | Selection |
| --- | --- | --- |
| hcf_fixed_first_fit.ini | fixed | first_feasible |
| hcf_max_gsnr_first_fit.ini | max_gsnr | first_feasible |
| hcf_minimum_power_first_fit.ini | minimum_required | first_feasible |
| hcf_minimum_power_ranked.ini | minimum_required | power_spectrum |

All four copy the active config's traffic, topology and physical parameters at
creation. They currently use C-band, 80 channels, 75 GHz, rates 100â€“600 Gbps,
3000 requests per iteration and one iteration. This is a development comparison;
use multiple matched seeds/iterations for paper results. Erlang loads run
sequentially (`thread_erlangs = False`) to avoid the previous host overload.
No run is started by installing these configs.

## Configuration contract

`hcf_power_objective` explicitly chooses `fixed`, `discrete`, `max_gsnr` or
`minimum_required`. It takes precedence over `optimized_power` within the
HCF allocation route. When absent, the existing behavior remains:
`optimized_power=True` selects max_gsnr; False selects the discrete grid.

- fixed: use exactly `input_power` watts (0.001 W = 0 dBm in these configs).
- discrete: evaluate `hcf_power_levels_dbm` as before.
- max_gsnr: retain the existing end-to-end maximum-GSNR optimizer.
- minimum_required: find the smallest common launch power on each directional
  path meeting mode threshold plus `hcf_margin_db`.

The fixed, max_gsnr and minimum_required objectives do not require a nonempty
discrete power grid. `resource_allocation_mode = hcf_power_opportunity` remains
the existing HCF controller entry point, including for the new probe-free policy.

## Minimum required power

The shared physical helper accumulates the existing model's coefficients:

    inverse_GSNR(P) = A/P + B*P^2 + C

A is the sum of span ASE coefficients; B is the sum of span nonlinear
coefficients; C includes IMI and the optional transceiver inverse SNR.
Partial spans, band/channel profiles and channel-center frequencies are reused.
For B > 0, P_peak = (A/(2B))^(1/3) brackets the decreasing branch of inverse
GSNR. If the peak cannot meet the target, reject the candidate. Otherwise
bisect that scalar branch, returning the feasible endpoint after at most 64
steps. Physical span coefficients are computed once per direction/mode/channel,
not at each bisection step. For B = 0, use A/(target_inverse - C), rejecting a
target at or above the asymptotic noise ceiling.

Forward and reverse directions can require different powers. The resource
ledger reserves their actual powers separately at each span/band amplifier,
and releases exactly those reservations. The existing scalar launch_power_w
field still reports the last directional power; candidate.demands is the
authoritative source for directional accounting.

No established connection is reoptimized or rechecked. This uses the existing
occupancy-independent GSNR model with its configured full-comb channel count.
The budgets constrain signal optical power, not electrical energy consumption.

## Selection policies

first_feasible retains path order and scans free channels in band/channel order
until a candidate satisfies GSNR and every amplifier budget.

power_spectrum evaluates ONLY the lowest-index common-free channel in the first
configured band with free spectrum on EACH candidate path. It checks eligible
modes and selects a budget-feasible candidate. A failed first channel does not
trigger a search of later channels/bands on that path. An entirely occupied
band can be skipped (for example, C full then first free L channel). Explicit
forced channel/band arguments retain their existing meanings.

Candidates across all paths are ranked by:

    J = alpha * sum_a(P_a / R_a) + (1-alpha) * sum_link(1 / F_link)

R_a is remaining amplifier signal-power headroom in watts; P_a is the actual
directional candidate demand. Count each directional amplifier/span once.
F_link counts wavelengths free in BOTH directions of a physical hop in the
candidate band; count the physical hop once. Channel continuity is enforced
separately by the common-free-channel helper. alpha is hcf_power_weight.
This is a congestion heuristic, not a prediction of future feasible requests.
The score uses sums, so long paths can incur larger costs; it is not the old
average power-scarcity score.

For power_spectrum, choose the lowest-score eligible mode on the selected
channel, then rank paths by score, scalar launch power, and routing index.
For minimum_required with first_feasible, choose the mode with the smallest
sum of amplifier demands on the first feasible channel.
Every candidate must satisfy all power budgets before selection.
Allocation revalidates spectrum and power, then commits via the existing
transactional resource lifecycle.

Neither probes nor virtual opportunity evaluations run under power_spectrum.
The old joint/first_channel_ranked policies retain their existing behavior.
With K paths and M eligible modes, at most K*M bidirectional physical
candidate evaluations are needed for an unforced power_spectrum request.
Each minimum-power calculation adds at most 64 inexpensive scalar iterations.
Numeric span/path coefficients and fixed/discrete GSNR values are reused in a
per-calculator LRU cache, capped at 16,384 entries and 16 MiB of accounted
key/value storage (not whole-process RSS). Keys include resolved physical
parameters. Iteration reset clears the cache. A second compact transmission-option
cache is bounded at 4,096 entries and 16 MiB, also cleared at iteration reset.
It stores only mode indices, directional powers and GSNR; physical/model settings
are in its key. Together the two caches have a 32 MiB accounted-storage cap
per allocator, not a process-RSS cap. Candidate demands, live resource
feasibility and scores are never retained. On a transmission-option cache miss, minimum-power bisection runs for the
requested threshold using reused coefficients; subsequent matching option
queries reuse its selected directional powers.

## Running

From E:/git/v6/FUSION, run one config at a time:

    .\.venv\Scripts\python.exe -m fusion.cli.run_sim run_sim --config_path .\fusion\configs\experiments\hcf_fixed_first_fit.ini --run_id 1
    .\.venv\Scripts\python.exe -m fusion.cli.run_sim run_sim --config_path .\fusion\configs\experiments\hcf_max_gsnr_first_fit.ini --run_id 1
    .\.venv\Scripts\python.exe -m fusion.cli.run_sim run_sim --config_path .\fusion\configs\experiments\hcf_minimum_power_first_fit.ini --run_id 1
    .\.venv\Scripts\python.exe -m fusion.cli.run_sim run_sim --config_path .\fusion\configs\experiments\hcf_minimum_power_ranked.ini --run_id 1

Keep load, request distribution, holding time, seeds, amplifier budgets and
band settings identical across comparisons. Compare blocking by bitrate/cause,
carried traffic, power occupancy and allocation time. Existing HCF metrics are
used; no measured improvement is claimed by this implementation.

The proposed method also changes the candidate search scope relative to
first_feasible. To isolate ranking from first-channel restriction, add a matched
ablation using the same first-channel-per-path candidates before making a
causal claim about ranking alone. A power_spectrum weight sweep (0, 0.5, 1)
isolates the influence of its two score terms.

