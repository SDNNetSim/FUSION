# Implementation Roadmap

This document outlines the phased implementation plan for FUSION GUI.

## Milestone Overview

| Milestone | Focus | Target |
|-----------|-------|--------|
| M1 | Core: Runs + Logs + Artifacts | Foundation |
| M2 | Progress + Config Editor | Usability |
| M3 | Network Viewer | Visualization |
| M4 | Polish + OSS Maturity | Production |

---

## Milestone 1: Core Infrastructure

**Goal**: Users can create runs, view logs, and download artifacts.

### Backend Tasks

| Task | Files | Description |
|------|-------|-------------|
| Project structure | `fusion/api/__init__.py`, `main.py`, `config.py` | FastAPI app setup, configuration |
| Database setup | `fusion/api/db/database.py`, `models.py` | SQLite + SQLAlchemy models |
| Run CRUD | `fusion/api/routes/runs.py`, `schemas/run.py` | Create, list, get, delete runs |
| Run manager service | `fusion/api/services/run_manager.py` | Subprocess management, process groups |
| Log streaming | `fusion/api/routes/runs.py` (SSE endpoint) | Tail sim.log via SSE |
| Artifact listing | `fusion/api/routes/artifacts.py` | List files in run directory |
| Artifact download | `fusion/api/routes/artifacts.py` | Secure file download |
| Config templates | `fusion/api/routes/configs.py` | List and get templates |
| Health endpoint | `fusion/api/routes/system.py` | Basic health check |
| CLI entry point | `fusion/cli/run_gui.py` | Launch uvicorn server |

**Endpoints (M1):**
```
POST   /api/runs                     Create run
GET    /api/runs                     List runs
GET    /api/runs/{id}                Get run
DELETE /api/runs/{id}                Cancel/delete run
GET    /api/runs/{id}/logs           SSE log stream
GET    /api/runs/{id}/artifacts      List artifacts
GET    /api/runs/{id}/artifacts/{p}  Download artifact
GET    /api/configs/templates        List templates
GET    /api/configs/templates/{name} Get template
GET    /api/health                   Health check
```

### Frontend Tasks

| Task | Files | Description |
|------|-------|-------------|
| Project setup | `frontend/package.json`, `vite.config.ts`, etc. | Vite + React + TypeScript + Tailwind |
| API client | `frontend/src/api/client.ts`, `runs.ts` | Axios wrapper, type-safe API calls |
| Layout | `frontend/src/components/layout/*` | Sidebar, header, main layout |
| UI primitives | `frontend/src/components/ui/*` | Button, Card, Badge (shadcn/ui) |
| Run list page | `frontend/src/pages/RunListPage.tsx` | Display all runs with status |
| Run card | `frontend/src/components/runs/RunCard.tsx` | Run summary card |
| Status badge | `frontend/src/components/runs/RunStatusBadge.tsx` | RUNNING, COMPLETED, etc. |
| New run page | `frontend/src/pages/NewRunPage.tsx` | Template selection, basic config |
| Run detail page | `frontend/src/pages/RunDetailPage.tsx` | Run info + logs + artifacts |
| Log viewer | `frontend/src/components/runs/LogViewer.tsx` | SSE-connected terminal-style viewer |
| File browser | `frontend/src/components/artifacts/FileBrowser.tsx` | Directory tree navigation |
| SSE hook | `frontend/src/hooks/useSSE.ts` | Generic SSE subscription |
| Run queries | `frontend/src/hooks/useRuns.ts` | TanStack Query hooks |

**UI Screens (M1):**
1. **Run List**: Grid/list of runs with status, actions
2. **New Run**: Template dropdown, name input, start button
3. **Run Detail**: Tabs for logs/artifacts, status header, cancel button

### Tests (M1)

**Backend:**
- `test_runs.py`: CRUD operations, status transitions
- `test_artifacts.py`: Path traversal prevention
- `test_health.py`: Health endpoint

**Frontend:**
- `RunCard.test.tsx`: Rendering, click handlers
- `LogViewer.test.tsx`: SSE connection, scroll behavior

### Docs Updates (M1)

- Complete `04-api.md` with M1 endpoints
- Add M1 setup instructions to `07-contributing.md`

### CI Gates (M1)

- All backend tests pass
- All frontend tests pass
- Lint checks pass
- Build succeeds

### Definition of Done (M1)

- [ ] User can run `fusion gui` and see the UI
- [ ] User can create a run from a template
- [ ] User can see real-time logs
- [ ] User can browse and download artifacts
- [ ] User can cancel a running simulation
- [ ] Completed runs survive server restart

---

## Milestone 2: Progress + Config Editor

**Goal**: Rich progress visualization and in-browser config editing.

### Backend Tasks

| Task | Files | Description |
|------|-------|-------------|
| Progress file support | CLI modification | Add `--progress-file` flag to simulator |
| Progress watcher | `fusion/api/services/progress_watcher.py` | Watch progress.jsonl, update DB |
| Progress SSE | `fusion/api/routes/runs.py` | Stream progress events |
| Config validation | `fusion/api/routes/configs.py` | Validate config without running |
| Config schema | `fusion/api/routes/configs.py` | Return JSON Schema for form generation |

**New Endpoints (M2):**
```
GET    /api/runs/{id}/progress       SSE progress stream
POST   /api/configs/validate         Validate config
GET    /api/configs/schema           Get JSON Schema
```

### Frontend Tasks

| Task | Files | Description |
|------|-------|-------------|
| Progress bar | `frontend/src/components/runs/ProgressBar.tsx` | Determinate progress indicator |
| Progress chart | `frontend/src/components/runs/ProgressChart.tsx` | Real-time blocking prob chart |
| Progress hook | `frontend/src/hooks/useProgress.ts` | SSE progress subscription |
| Config editor page | `frontend/src/pages/ConfigEditorPage.tsx` | Full config editing |
| Config text editor | `frontend/src/components/config/ConfigTextEditor.tsx` | INI syntax highlighting |
| Config validation | `frontend/src/components/config/ValidationErrors.tsx` | Show validation errors |
| Log search | `frontend/src/components/runs/LogViewer.tsx` | Search/filter in logs |
| Log follow toggle | `frontend/src/components/runs/LogViewer.tsx` | Auto-scroll toggle |
| Dark mode | `frontend/src/stores/ui.ts`, theme provider | Theme switching |

**UI Screens (M2):**
1. **Run Detail (enhanced)**: Progress bar, metrics chart, ETA
2. **Config Editor**: Text editor with validation, save as template
3. **Settings**: Theme toggle

### Tests (M2)

**Backend:**
- `test_progress.py`: Progress parsing, SSE events
- `test_config_validation.py`: Various invalid configs

**Frontend:**
- `ProgressBar.test.tsx`: Value updates
- `ConfigTextEditor.test.tsx`: Syntax errors

**E2E (Playwright) - Required for M2:**
- `e2e/create-run.spec.ts`: Create run -> observe logs -> cancel
- `e2e/view-artifacts.spec.ts`: Create run -> view artifacts -> download

See [10-testing.md](10-testing.md) for details on fake simulator mode for fast E2E.

### Docs Updates (M2)

- Update `03-run-directory-contract.md` with progress.jsonl schema
- Document config validation in `04-api.md`

### CI Gates (M2)

- Progress integration test (run simulation, verify progress events)
- Config validation tests
- E2E tests passing (Playwright)

### Definition of Done (M2)

- [ ] Progress bar shows during simulation
- [ ] Real-time chart shows blocking probability
- [ ] User can edit config in browser
- [ ] Validation errors shown inline
- [ ] Dark mode works
- [ ] Log viewer has search and follow toggle
- [ ] 2 E2E tests passing (create/cancel run, view/download artifacts)

---

## Milestone 3: Network Viewer

**Goal**: Visualize network topology with utilization data.

### Backend Tasks

| Task | Files | Description |
|------|-------|-------------|
| Topology listing | `fusion/api/routes/topology.py` | List available topologies |
| Topology data | `fusion/api/routes/topology.py` | Get nodes/links with coordinates |
| Run topology state | `fusion/api/routes/topology.py` | Topology with utilization from run |

**New Endpoints (M3):**
```
GET    /api/topology                 List topologies
GET    /api/topology/{name}          Get topology data
GET    /api/runs/{id}/topology       Get topology with run utilization
```

### Frontend Tasks

| Task | Files | Description |
|------|-------|-------------|
| Topology page | `frontend/src/pages/TopologyPage.tsx` | Standalone topology viewer |
| Network graph | `frontend/src/components/topology/NetworkGraph.tsx` | D3/React Flow visualization |
| Node component | `frontend/src/components/topology/NetworkNode.tsx` | Node rendering |
| Link component | `frontend/src/components/topology/NetworkLink.tsx` | Link with utilization color |
| Node tooltip | `frontend/src/components/topology/NodeTooltip.tsx` | Hover details |
| Path highlight | `frontend/src/components/topology/NetworkGraph.tsx` | Highlight selected path |
| Utilization legend | `frontend/src/components/topology/UtilizationLegend.tsx` | Color scale legend |
| Run topology tab | `frontend/src/pages/RunDetailPage.tsx` | Add topology tab to run detail |

**UI Screens (M3):**
1. **Topology Page**: Select topology, view graph
2. **Run Detail Topology Tab**: Graph with real-time utilization

### Tests (M3)

**Backend:**
- `test_topology.py`: Topology data format

**Frontend:**
- `NetworkGraph.test.tsx`: Renders with mock data
- Visual regression tests (optional)

### Docs Updates (M3)

- Add topology endpoints to `04-api.md`
- Document visualization features

### Definition of Done (M3)

- [ ] User can view any topology as a graph
- [ ] Nodes and links are labeled
- [ ] Utilization shown as link color
- [ ] Can view topology during/after run
- [ ] Pan and zoom work

---

## Milestone 4: Polish + OSS Maturity

**Goal**: Production quality, excellent DX, ready for public use.

### Backend Tasks

| Task | Files | Description |
|------|-------|-------------|
| Rate limiting | `fusion/api/main.py` | Optional rate limiting |
| Request logging | `fusion/api/main.py` | Structured request logs |
| Error handling | `fusion/api/main.py` | Consistent error responses |
| OpenAPI improvements | `fusion/api/main.py` | Better descriptions, examples |
| Concurrent runs | `fusion/api/services/run_manager.py` | Configurable concurrency |

### Frontend Tasks

| Task | Files | Description |
|------|-------|-------------|
| Error boundaries | `frontend/src/components/ErrorBoundary.tsx` | Graceful error handling |
| Loading skeletons | `frontend/src/components/ui/Skeleton.tsx` | Better loading states |
| Keyboard shortcuts | Global | Common actions (Ctrl+N, etc.) |
| Responsive design | Various | Mobile/tablet support |
| Onboarding | `frontend/src/components/Onboarding.tsx` | First-use guidance |
| Run comparison | `frontend/src/pages/CompareRunsPage.tsx` | Compare multiple runs |
| Export results | `frontend/src/components/runs/ExportButton.tsx` | Export to CSV/JSON |

### Documentation Tasks

| Task | Files | Description |
|------|-------|-------------|
| User guide | `docs/gui/user-guide.md` | End-user documentation |
| API examples | `docs/gui/04-api.md` | cURL examples for all endpoints |
| Troubleshooting | `docs/gui/troubleshooting.md` | Common issues and solutions |
| Issue templates | `.github/ISSUE_TEMPLATE/` | Bug report, feature request |
| PR template | `.github/PULL_REQUEST_TEMPLATE.md` | PR checklist |

### Tests (M4)

**Additional E2E Tests:**
- `e2e/config-editor.spec.ts`: Edit config, validate, save
- `e2e/topology-viewer.spec.ts`: View topology, check utilization
- `e2e/run-comparison.spec.ts`: Compare two runs side by side

**Non-functional Tests:**
- Performance: API response time < 100ms (p95)
- Accessibility: axe-core audit

### Definition of Done (M4)

- [ ] No known bugs
- [ ] User guide complete
- [ ] Issue templates in place
- [ ] All E2E tests passing (5+ total)
- [ ] Performance acceptable (< 100ms API responses)
- [ ] Accessibility audit passed
- [ ] Cross-browser tested (Chrome, Firefox, Safari)

---

## Future Milestones (Post-MVP)

### M5: HPC Integration

- Remote job submission (SSH + SLURM)
- Queue status monitoring
- Remote artifact fetching
- Cluster configuration UI

### M6: Advanced Analytics

- Run comparison dashboard
- Batch job management
- Custom plot generation
- Report export (PDF)

### M7: Collaboration (if needed)

- User accounts
- Shared runs
- Comments/annotations
- Team workspaces

---

## Branch Strategy by Milestone

```
main ─────────────────────────────────────────────────────►
                     │              │             │
                   v6.1.0        v6.2.0        v6.3.0
                     │              │             │
develop ─┬───────────┴──────────────┴─────────────┴───────►
         │                 │                │
    feature/gui-m1    feature/gui-m2   feature/gui-m3
```

Each milestone:
1. Branch `feature/gui-mN` from `develop`
2. Complete all tasks in milestone
3. PR to `develop`
4. After stabilization, merge `develop` to `main`
5. Tag release

## Simulator Modifications Required

The following changes to the core simulator are needed:

### M1 (Required)

None - use existing CLI as-is.

### M2 (Required)

Add `--progress-file` CLI argument:

```python
# fusion/cli/run_sim.py
parser.add_argument(
    "--progress-file",
    type=str,
    help="Path to write progress events (JSONL format)"
)
```

Modify `SimulationEngine` to write progress events:

```python
# fusion/core/simulation.py
def end_iter(self, iteration: int, ...):
    # Existing code...

    # Write progress event
    if self.progress_file:
        event = {
            "type": "iteration",
            "ts": datetime.utcnow().isoformat(),
            "erlang": self.engine_props["erlang"],
            "iteration": iteration,
            "total_iterations": self.engine_props["max_iters"],
            "metrics": {
                "blocking_prob": blocking_mean,
            }
        }
        with open(self.progress_file, "a") as f:
            f.write(json.dumps(event) + "\n")
```

### M3 (Required)

Add topology coordinate data or use algorithmic layout:

```python
# fusion/api/routes/topology.py
# Either:
# 1. Store coordinates in topology JSON files
# 2. Use force-directed layout algorithm
# 3. Use geographic coordinates if available
```
