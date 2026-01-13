# ADR-0007: SDNOrchestrator Design

## Status

Accepted

## Context

The legacy `SDNController` has grown into a monolith containing:
- Routing algorithm logic
- Spectrum assignment logic
- Grooming calculations
- Slicing decisions
- SNR validation
- Protection handling
- RL-specific code paths (`mock_handle_arrival`)

This leads to:
1. **Code duplication**: RL and simulation have separate feasibility checks
2. **Feature explosion**: Adding features requires modifying SDNController
3. **Testing difficulty**: Cannot test features in isolation
4. **Maintenance burden**: Understanding the flow requires reading 1000+ lines

## Decision

We will create a **thin orchestrator** (`SDNOrchestrator`) that:
1. **Routes** requests through independent pipelines
2. **Coordinates** pipeline execution order
3. **Combines** results into a single `AllocationResult`
4. **Does NOT** implement any algorithm logic

### Rules for SDNOrchestrator

#### ALLOWED

| Category | Examples |
|----------|----------|
| Stage sequencing | `if self.grooming: groom_result = self.grooming.try_groom(...)` |
| Feature checking | `if self.config.grooming_enabled` |
| Result combination | Merging groomed + allocated lightpaths |
| Rollback coordination | `network_state.release_lightpath(...)` on failure |
| Error handling | Catching exceptions, returning `BlockReason` |

#### FORBIDDEN

| Category | Examples | Why |
|----------|----------|-----|
| Algorithm logic | K-shortest-path, first-fit, SNR calculation | Belongs in pipelines |
| Feature-specific logic | Grooming bandwidth calculations | Belongs in GroomingPipeline |
| Data structure access | `cores_matrix`, `lightpath_status_dict` | NetworkState encapsulates these |
| Value-based branching | `if modulation == "QPSK"` | Configuration, not orchestration |
| State caching | `self._cached_routes` | Fresh state each call |

### Size Constraints

| Metric | Limit |
|--------|-------|
| Total lines in orchestrator.py | < 200 |
| Lines per method | < 50 |
| Lines added per feature | < 20 |

## Consequences

### Positive

1. **Single code path**: RL and simulation use the same pipelines
2. **Testable**: Each pipeline can be tested independently
3. **Extensible**: New features add a pipeline, not modify orchestrator
4. **Maintainable**: Orchestrator remains small and readable
5. **Composable**: Features combine without special cases

### Negative

1. **More files**: Each feature needs its own pipeline file
2. **Indirection**: Flow spans multiple files
3. **Learning curve**: Understanding the pattern takes time

### Neutral

1. **Migration effort**: Existing SDNController code moves to pipelines
2. **Performance**: Minimal overhead from additional method calls

## Alternatives Considered

### 1. Refactor SDNController in Place

**Rejected**: Would not solve the fundamental problem of mixing orchestration with algorithms.

### 2. Multiple Specialized SDNControllers

**Rejected**: Would still duplicate common orchestration logic.

### 3. Strategy Pattern Only

**Rejected**: Does not address the coordination and result combination needs.

## Implementation

See `migration/phase_3_orchestrator.md` for implementation details.

## PR Review Checklist

When reviewing PRs that modify `orchestrator.py`:

```markdown
## SDNOrchestrator Change Checklist

### Size Check
- [ ] orchestrator.py total lines < 200
- [ ] No method > 50 lines
- [ ] This PR adds < 20 lines to orchestrator

### Logic Check
- [ ] No algorithm implementations (sorting, searching, calculations)
- [ ] No direct numpy array access
- [ ] No hardcoded thresholds or magic numbers
- [ ] No feature-specific conditionals beyond "is feature enabled?"

### Delegation Check
- [ ] New functionality is in a pipeline, not inline
- [ ] Pipeline created by PipelineFactory
- [ ] Pipeline has its own unit tests

### Interface Check
- [ ] Uses standard result types (RouteResult, SpectrumResult, etc.)
- [ ] Passes Request and NetworkState, not dicts
- [ ] Returns AllocationResult

### Rollback Check
- [ ] Failure paths call release_lightpath() appropriately
- [ ] Partial allocation handled correctly
- [ ] No resource leaks on exceptions
```

## Examples

### Good Change: Adding QoS Pipeline

```python
# PipelineSet gets new field
@dataclass
class PipelineSet:
    routing: RoutingPipeline
    spectrum: SpectrumPipeline
    qos: QoSPipeline | None = None  # NEW

# Orchestrator gets 3 lines
def handle_arrival(self, ...):
    route_result = self.routing.find_routes(...)

    if self.qos and self.config.qos_enabled:
        route_result = self.qos.prioritize(route_result, request)

    # rest unchanged
```

**Why good**: Algorithm logic in QoSPipeline, orchestrator only coordinates.

### Bad Change: Inline QoS Logic

```python
# BAD: Algorithm logic in orchestrator
def handle_arrival(self, ...):
    route_result = self.routing.find_routes(...)

    if self.config.qos_enabled:
        priority = request.qos_class
        if priority == "gold":
            route_result.paths = [route_result.paths[0]]
        elif priority == "silver":
            route_result.paths.sort(key=lambda p: self._compute_congestion(p))
        # ... more algorithm logic
```

**Why bad**: QoS algorithm logic belongs in a pipeline, not orchestrator.

## Related Decisions

- ADR-0002: NetworkState as single source of truth
- ADR-0003: Result object design
- ADR-0006: NetworkState authority

## References

- `architecture/orchestration.md` - Full orchestrator documentation
- `architecture/pipelines.md` - Pipeline implementations
- `architecture/pipeline_interfaces.md` - Protocol definitions
