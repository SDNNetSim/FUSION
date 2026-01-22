# Testing Strategy

This document defines the testing strategy for FUSION GUI, including the testing pyramid, minimum test requirements for MVP, and details on the fake simulator mode.

## Testing Pyramid

```
                    /\
                   /  \
                  / E2E \           <- 2+ tests (Playwright)
                 /------\
                /        \
               / Integration\       <- API route tests (pytest)
              /--------------\
             /                \
            /    Unit Tests    \    <- Components + services (Vitest/pytest)
           /--------------------\
```

**Test distribution for MVP:**

| Layer | Tool | Count (MVP) | Focus |
|-------|------|-------------|-------|
| Unit (Backend) | pytest | 10+ | RunManager, services, utilities |
| Unit (Frontend) | Vitest | 10+ | Components, hooks, utilities |
| Integration | pytest | 5+ | API routes with test database |
| E2E | Playwright | 2+ | Critical user flows |

---

## Testing Ladder (Frontend)

Use the right test type for each scenario:

| Level | Tool | When to Use | Speed |
|-------|------|-------------|-------|
| **Unit** | Vitest | Pure functions, utilities, hooks without DOM | Fast |
| **Component** | Vitest + RTL | Isolated component rendering, props, events | Fast |
| **Integration** | Vitest + RTL + MSW | Components with API calls, multi-component flows | Medium |
| **E2E** | Playwright + fake simulator | Full user journeys, critical paths | Slow |

### PR Test Expectations

| PR Type | Required Tests |
|---------|----------------|
| **UI-only** (new component) | Component test with RTL |
| **Backend-only** (new endpoint) | pytest route + service tests |
| **API contract change** | Backend pytest + frontend MSW handler update + integration test |
| **Bug fix** | Regression test covering the bug |
| **Critical flow change** | E2E test update or addition |

### What NOT to Test

- Third-party library internals (shadcn/ui, TanStack Query)
- Styling details (use visual regression if needed, not unit tests)
- Implementation details (test behavior, not internal state)

---

## Accessibility Testing

Run accessibility checks as part of E2E tests using `@axe-core/playwright`.

### Setup

```bash
cd frontend && npm install -D @axe-core/playwright
```

### Smoke Test

Add to at least one E2E test per major page:

```typescript
// frontend/e2e/accessibility.spec.ts
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("run list page has no critical a11y violations", async ({ page }) => {
  await page.goto("/");
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();

  expect(results.violations.filter((v) => v.impact === "critical")).toEqual([]);
});
```

### CI Gate

Accessibility tests are **advisory in M2**, **blocking in M4**. Critical violations (missing labels, broken focus) should block PRs.

---

## Flake Policy

E2E tests can be flaky due to timing, animations, or network variability.

**CI Configuration:**
- Playwright retries: **1** (not more)
- If a test passes only on retry, it is considered a flake

**Handling Flakes:**
1. First occurrence: Note in PR, investigate root cause
2. Repeated flakes (2+ in a week): Open a tracking issue labeled `flaky-test`
3. Persistent flakes: Either fix or quarantine the test (skip with `test.skip` + issue link)

**Common Fixes:**
- Add explicit `waitFor` instead of arbitrary timeouts
- Use `toBeVisible()` before interacting
- Ensure fake simulator determinism via fixed `FAKE_SIM_DURATION`

---

## Backend Testing (pytest)

### Test Structure

```
fusion/api/tests/
├── conftest.py              # Shared fixtures
├── test_runs.py             # Run CRUD and lifecycle
├── test_artifacts.py        # Artifact listing and download
├── test_configs.py          # Config templates and validation
├── test_health.py           # Health endpoint
└── test_run_manager.py      # RunManager unit tests
```

### Fixtures (conftest.py)

```python
# fusion/api/tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fusion.api.main import app
from fusion.api.db.database import get_db, Base
from fusion.api.db.models import Run


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """Create a test client with overridden database dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_run(db_session) -> Run:
    """Create a sample run for testing."""
    run = Run(
        id="test-run-001",
        name="Test Run",
        config_json='{"template": "test"}',
        status="PENDING",
    )
    db_session.add(run)
    db_session.commit()
    return run


@pytest.fixture
def run_with_artifacts(tmp_path, sample_run):
    """Create a run with artifact files."""
    run_dir = tmp_path / "gui_runs" / sample_run.id
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)

    # Create sample artifacts
    (output_dir / "results.json").write_text('{"blocking_prob": 0.05}')
    (output_dir / "sim.log").write_text("Simulation started\n")

    sample_run.output_dir = str(run_dir)
    return sample_run, run_dir
```

### Fake Simulator

For testing run lifecycle without real simulations, use a fake simulator script.

**Important:** The fake simulator lives in `fusion/api/devtools/`, not `fusion/api/tests/`. Test-only folders (`tests/`) may be excluded from wheels and should never be invoked at runtime. The `devtools/` module is packaged and safe for runtime use when `FUSION_GUI_FAKE_SIMULATOR=true`.

```python
# fusion/api/devtools/fake_simulator.py
#!/usr/bin/env python3
"""
Fake simulator for testing.

Usage:
    python fake_simulator.py --config config.ini --output-dir /path/to/output

Behavior controlled by environment variables:
    FAKE_SIM_DURATION: Sleep duration in seconds (default: 0.1)
    FAKE_SIM_EXIT_CODE: Exit code to return (default: 0)
    FAKE_SIM_FAIL_AFTER: Fail after N iterations (default: never)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-file", default=None)
    args = parser.parse_args()

    duration = float(os.environ.get("FAKE_SIM_DURATION", "0.1"))
    exit_code = int(os.environ.get("FAKE_SIM_EXIT_CODE", "0"))
    fail_after = int(os.environ.get("FAKE_SIM_FAIL_AFTER", "-1"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir.parent / "logs" / "sim.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    total_iterations = 5
    erlangs = [10, 20, 30]

    with open(log_file, "w") as log:
        log.write(f"[{datetime.now().isoformat()}] Simulation started\n")
        log.flush()

        for erlang in erlangs:
            for iteration in range(1, total_iterations + 1):
                # Check for failure trigger
                total_iters_done = erlangs.index(erlang) * total_iterations + iteration
                if fail_after > 0 and total_iters_done >= fail_after:
                    log.write(f"[{datetime.now().isoformat()}] ERROR: Simulated failure\n")
                    sys.exit(1)

                time.sleep(duration / (len(erlangs) * total_iterations))

                log.write(
                    f"[{datetime.now().isoformat()}] "
                    f"Erlang {erlang}, Iteration {iteration}/{total_iterations}\n"
                )
                log.flush()

                # Write progress event
                if args.progress_file:
                    with open(args.progress_file, "a") as pf:
                        event = {
                            "type": "iteration",
                            "ts": datetime.now().isoformat(),
                            "erlang": erlang,
                            "iteration": iteration,
                            "total_iterations": total_iterations,
                            "metrics": {"blocking_prob": 0.05 * iteration / total_iterations},
                        }
                        pf.write(json.dumps(event) + "\n")

            # Write erlang result
            result_file = output_dir / f"{erlang}_erlang.json"
            result_file.write_text(json.dumps({
                "erlang": erlang,
                "blocking_prob": 0.05,
                "iterations": total_iterations,
            }))

        log.write(f"[{datetime.now().isoformat()}] Simulation completed\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

### RunManager Unit Tests

```python
# fusion/api/tests/test_run_manager.py
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from fusion.api.services.run_manager import RunManager


@pytest.fixture
def run_manager(tmp_path):
    """Create RunManager with temp directory."""
    return RunManager(runs_dir=tmp_path / "gui_runs")


@pytest.fixture
def fake_simulator_path():
    """Path to fake simulator script (in devtools, not tests)."""
    return Path(__file__).parent.parent / "devtools" / "fake_simulator.py"


class TestRunManager:
    """Unit tests for RunManager."""

    def test_create_run_creates_directory_structure(self, run_manager):
        """Creating a run should set up the expected directories."""
        run_id = run_manager.create_run(
            name="Test",
            config={"template": "test"},
        )

        run_dir = run_manager.runs_dir / run_id
        assert run_dir.exists()
        assert (run_dir / "logs").exists()
        assert (run_dir / "output").exists()
        assert (run_dir / "config.ini").exists()

    def test_create_run_writes_config(self, run_manager):
        """Config should be written to run directory."""
        run_id = run_manager.create_run(
            name="Test",
            config={"erlang_start": 10, "erlang_end": 50},
        )

        config_path = run_manager.runs_dir / run_id / "config.ini"
        assert config_path.exists()
        content = config_path.read_text()
        assert "erlang_start" in content or "10" in content

    @patch("fusion.api.services.run_manager.subprocess.Popen")
    def test_start_run_launches_subprocess(self, mock_popen, run_manager):
        """Starting a run should launch simulator subprocess."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        run_id = run_manager.create_run(name="Test", config={})
        run_manager.start_run(run_id)

        mock_popen.assert_called_once()
        call_kwargs = mock_popen.call_args.kwargs
        assert call_kwargs.get("start_new_session", False) or os.name == "nt"

    @patch("fusion.api.services.run_manager.subprocess.Popen")
    def test_start_run_stores_pid(self, mock_popen, run_manager, db_session):
        """PID should be stored for later cancellation."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        run_id = run_manager.create_run(name="Test", config={})
        run_manager.start_run(run_id)

        # Verify PID is tracked
        assert run_manager.get_run_pid(run_id) == 12345

    def test_cancel_run_not_running_returns_false(self, run_manager):
        """Cancelling a non-running run should return False."""
        run_id = run_manager.create_run(name="Test", config={})
        result = run_manager.cancel_run(run_id)
        assert result is False

    @pytest.mark.skipif(os.name == "nt", reason="POSIX-specific test")
    @patch("os.killpg")
    @patch("os.getpgid")
    def test_cancel_run_kills_process_group_posix(
        self, mock_getpgid, mock_killpg, run_manager
    ):
        """On POSIX, cancellation should kill the process group."""
        import signal

        mock_getpgid.return_value = 12345

        run_id = run_manager.create_run(name="Test", config={})
        run_manager._running_processes[run_id] = MagicMock(pid=12345)

        run_manager.cancel_run(run_id)

        mock_killpg.assert_called_with(12345, signal.SIGTERM)

    def test_get_log_path_returns_correct_path(self, run_manager):
        """Log path should be within run directory."""
        run_id = run_manager.create_run(name="Test", config={})
        log_path = run_manager.get_log_path(run_id)

        expected = run_manager.runs_dir / run_id / "logs" / "sim.log"
        assert log_path == expected


class TestRunManagerIntegration:
    """Integration tests using fake simulator."""

    @pytest.mark.slow
    def test_run_completes_successfully(self, run_manager, fake_simulator_path):
        """Full run lifecycle with fake simulator."""
        with patch.dict(os.environ, {"FAKE_SIM_DURATION": "0.1"}):
            run_id = run_manager.create_run(name="Test", config={})
            run_manager.start_run(run_id, simulator_cmd=[
                sys.executable, str(fake_simulator_path)
            ])

            # Wait for completion (with timeout)
            import time
            for _ in range(50):  # 5 second timeout
                if run_manager.get_run_status(run_id) == "COMPLETED":
                    break
                time.sleep(0.1)

            assert run_manager.get_run_status(run_id) == "COMPLETED"

    @pytest.mark.slow
    def test_run_failure_detected(self, run_manager, fake_simulator_path):
        """Failed runs should be marked as FAILED."""
        with patch.dict(os.environ, {
            "FAKE_SIM_DURATION": "0.1",
            "FAKE_SIM_EXIT_CODE": "1",
        }):
            run_id = run_manager.create_run(name="Test", config={})
            run_manager.start_run(run_id, simulator_cmd=[
                sys.executable, str(fake_simulator_path)
            ])

            import time
            for _ in range(50):
                status = run_manager.get_run_status(run_id)
                if status in ("COMPLETED", "FAILED"):
                    break
                time.sleep(0.1)

            assert run_manager.get_run_status(run_id) == "FAILED"
```

### API Route Tests

```python
# fusion/api/tests/test_runs.py
import pytest


class TestRunsAPI:
    """Tests for /api/runs endpoints."""

    def test_create_run_returns_201(self, client):
        """POST /api/runs should return 201 with run ID."""
        response = client.post("/api/runs", json={
            "name": "Test Run",
            "template": "default",
            "config": {},
        })
        assert response.status_code == 201
        assert "id" in response.json()

    def test_create_run_invalid_template_returns_400(self, client):
        """Invalid template should return 400."""
        response = client.post("/api/runs", json={
            "name": "Test",
            "template": "nonexistent_template",
            "config": {},
        })
        assert response.status_code == 400

    def test_list_runs_returns_array(self, client):
        """GET /api/runs should return array."""
        response = client.get("/api/runs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_runs_with_status_filter(self, client, sample_run):
        """Status filter should work."""
        response = client.get("/api/runs?status=PENDING")
        assert response.status_code == 200
        runs = response.json()
        assert all(r["status"] == "PENDING" for r in runs)

    def test_get_run_returns_details(self, client, sample_run):
        """GET /api/runs/{id} should return run details."""
        response = client.get(f"/api/runs/{sample_run.id}")
        assert response.status_code == 200
        assert response.json()["id"] == sample_run.id

    def test_get_run_not_found(self, client):
        """Non-existent run should return 404."""
        response = client.get("/api/runs/nonexistent")
        assert response.status_code == 404

    def test_delete_run_cancels_if_running(self, client, sample_run, db_session):
        """DELETE should cancel running run."""
        sample_run.status = "RUNNING"
        db_session.commit()

        response = client.delete(f"/api/runs/{sample_run.id}")
        assert response.status_code == 200

        # Verify status changed
        db_session.refresh(sample_run)
        assert sample_run.status == "CANCELLED"
```

### Artifact Security Tests

```python
# fusion/api/tests/test_artifacts.py
import pytest
from pathlib import Path


class TestArtifactsAPI:
    """Tests for /api/runs/{id}/artifacts endpoints."""

    def test_list_artifacts_returns_files(self, client, run_with_artifacts):
        """GET /api/runs/{id}/artifacts should list files."""
        run, _ = run_with_artifacts
        response = client.get(f"/api/runs/{run.id}/artifacts")
        assert response.status_code == 200
        artifacts = response.json()
        assert any(a["name"] == "results.json" for a in artifacts)

    def test_download_artifact_returns_file(self, client, run_with_artifacts):
        """GET /api/runs/{id}/artifacts/{path} should return file content."""
        run, _ = run_with_artifacts
        response = client.get(f"/api/runs/{run.id}/artifacts/output/results.json")
        assert response.status_code == 200
        assert response.json()["blocking_prob"] == 0.05

    def test_path_traversal_blocked(self, client, run_with_artifacts):
        """Path traversal attempts should be rejected."""
        run, _ = run_with_artifacts

        # Various traversal attempts
        traversal_paths = [
            "../../../etc/passwd",
            "..%2F..%2Fetc/passwd",
            "output/../../../etc/passwd",
            "/etc/passwd",
        ]

        for path in traversal_paths:
            response = client.get(f"/api/runs/{run.id}/artifacts/{path}")
            assert response.status_code == 403, f"Path {path} should be blocked"

    def test_symlink_within_run_allowed(self, client, run_with_artifacts, tmp_path):
        """Symlinks pointing within run directory should work."""
        run, run_dir = run_with_artifacts

        # Create symlink within run directory
        link_path = run_dir / "link_to_results.json"
        link_path.symlink_to(run_dir / "output" / "results.json")

        response = client.get(f"/api/runs/{run.id}/artifacts/link_to_results.json")
        assert response.status_code == 200

    def test_symlink_escape_blocked(self, client, run_with_artifacts, tmp_path):
        """Symlinks pointing outside run directory should be rejected."""
        run, run_dir = run_with_artifacts

        # Create file outside run directory
        external_file = tmp_path / "external_secret.txt"
        external_file.write_text("secret data")

        # Create symlink pointing outside
        escape_link = run_dir / "escape.txt"
        escape_link.symlink_to(external_file)

        response = client.get(f"/api/runs/{run.id}/artifacts/escape.txt")
        assert response.status_code == 403

    def test_nonexistent_artifact_returns_404(self, client, run_with_artifacts):
        """Non-existent file should return 404."""
        run, _ = run_with_artifacts
        response = client.get(f"/api/runs/{run.id}/artifacts/nonexistent.txt")
        assert response.status_code == 404
```

---

## Frontend Testing (Vitest + Testing Library)

### Test Structure

```
frontend/src/
├── components/
│   ├── runs/
│   │   ├── RunCard.tsx
│   │   ├── RunCard.test.tsx
│   │   ├── LogViewer.tsx
│   │   ├── LogViewer.test.tsx
│   │   ├── RunStatusBadge.tsx
│   │   └── RunStatusBadge.test.tsx
│   └── artifacts/
│       ├── FileBrowser.tsx
│       └── FileBrowser.test.tsx
├── hooks/
│   ├── useSSE.ts
│   ├── useSSE.test.ts
│   ├── useRuns.ts
│   └── useRuns.test.ts
└── test/
    ├── setup.ts           # Test setup
    ├── mocks/             # Mock implementations
    │   ├── handlers.ts    # MSW handlers
    │   └── server.ts      # MSW server
    └── utils.tsx          # Test utilities
```

### Test Setup

```typescript
// frontend/src/test/setup.ts
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeAll, afterAll } from "vitest";
import { server } from "./mocks/server";

// Start MSW server
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterAll(() => server.close());
afterEach(() => {
  cleanup();
  server.resetHandlers();
});
```

```typescript
// frontend/src/test/mocks/handlers.ts
import { http, HttpResponse } from "msw";

export const handlers = [
  // List runs
  http.get("/api/runs", () => {
    return HttpResponse.json([
      {
        id: "run-1",
        name: "Test Run 1",
        status: "COMPLETED",
        created_at: "2024-01-01T00:00:00Z",
      },
      {
        id: "run-2",
        name: "Test Run 2",
        status: "RUNNING",
        created_at: "2024-01-02T00:00:00Z",
      },
    ]);
  }),

  // Get single run
  http.get("/api/runs/:id", ({ params }) => {
    return HttpResponse.json({
      id: params.id,
      name: `Run ${params.id}`,
      status: "RUNNING",
      created_at: "2024-01-01T00:00:00Z",
    });
  }),

  // Create run
  http.post("/api/runs", async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json(
      { id: "new-run-123", ...body, status: "PENDING" },
      { status: 201 }
    );
  }),

  // List artifacts
  http.get("/api/runs/:id/artifacts", () => {
    return HttpResponse.json([
      { name: "output", type: "directory", size: 0 },
      { name: "sim.log", type: "file", size: 1024 },
    ]);
  }),

  // Health check
  http.get("/api/health", () => {
    return HttpResponse.json({ status: "healthy" });
  }),
];
```

```typescript
// frontend/src/test/mocks/server.ts
import { setupServer } from "msw/node";
import { handlers } from "./handlers";

export const server = setupServer(...handlers);
```

### Test Utilities

```text
// frontend/src/test/utils.tsx
import { ReactElement } from "react";
import { render, RenderOptions } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

interface WrapperProps {
  children: React.ReactNode;
}

function AllProviders({ children }: WrapperProps) {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  );
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, "wrapper">
) => render(ui, { wrapper: AllProviders, ...options });

export * from "@testing-library/react";
export { customRender as render };
```

### Component Tests

```typescript
// frontend/src/components/runs/RunCard.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "../../test/utils";
import { RunCard } from "./RunCard";

const mockRun = {
  id: "test-run-1",
  name: "Test Simulation",
  status: "RUNNING" as const,
  created_at: "2024-01-15T10:30:00Z",
  progress: { current: 5, total: 10 },
};

describe("RunCard", () => {
  it("renders run name", () => {
    render(<RunCard run={mockRun} />);
    expect(screen.getByText("Test Simulation")).toBeInTheDocument();
  });

  it("displays correct status badge", () => {
    render(<RunCard run={mockRun} />);
    expect(screen.getByText("RUNNING")).toBeInTheDocument();
  });

  it("shows progress when running", () => {
    render(<RunCard run={mockRun} />);
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();
  });

  it("calls onClick when clicked", () => {
    const onClick = vi.fn();
    render(<RunCard run={mockRun} onClick={onClick} />);

    fireEvent.click(screen.getByRole("article"));
    expect(onClick).toHaveBeenCalledWith(mockRun.id);
  });

  it("shows cancel button for running runs", () => {
    const onCancel = vi.fn();
    render(<RunCard run={mockRun} onCancel={onCancel} />);

    const cancelBtn = screen.getByRole("button", { name: /cancel/i });
    fireEvent.click(cancelBtn);
    expect(onCancel).toHaveBeenCalledWith(mockRun.id);
  });

  it("hides cancel button for completed runs", () => {
    const completedRun = { ...mockRun, status: "COMPLETED" as const };
    render(<RunCard run={completedRun} />);

    expect(screen.queryByRole("button", { name: /cancel/i })).not.toBeInTheDocument();
  });
});
```

```typescript
// frontend/src/components/runs/LogViewer.test.tsx
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "../../test/utils";
import { LogViewer } from "./LogViewer";

// Mock EventSource
class MockEventSource {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onopen: ((event: Event) => void) | null = null;
  readyState = 0;

  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = 1;
      this.onopen?.(new Event("open"));
    }, 0);
  }

  close = vi.fn();

  // Test helper to simulate messages
  simulateMessage(data: string) {
    this.onmessage?.(new MessageEvent("message", { data }));
  }

  simulateError() {
    this.readyState = 2;
    this.onerror?.(new Event("error"));
  }
}

describe("LogViewer", () => {
  let mockEventSource: MockEventSource;

  beforeEach(() => {
    vi.stubGlobal("EventSource", vi.fn((url: string) => {
      mockEventSource = new MockEventSource(url);
      return mockEventSource;
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("connects to SSE endpoint", () => {
    render(<LogViewer runId="test-123" />);

    expect(EventSource).toHaveBeenCalledWith(
      expect.stringContaining("/api/runs/test-123/logs")
    );
  });

  it("displays log lines as they arrive", async () => {
    render(<LogViewer runId="test-123" />);

    await waitFor(() => {
      mockEventSource.simulateMessage("First log line");
    });

    expect(screen.getByText("First log line")).toBeInTheDocument();
  });

  it("auto-scrolls when follow is enabled", async () => {
    const { container } = render(<LogViewer runId="test-123" follow={true} />);
    const logContainer = container.querySelector("[data-testid='log-container']");

    await waitFor(() => {
      mockEventSource.simulateMessage("New line");
    });

    // Verify scroll behavior (simplified check)
    expect(logContainer?.scrollTop).toBeDefined();
  });

  it("shows reconnecting state on error", async () => {
    render(<LogViewer runId="test-123" />);

    await waitFor(() => {
      mockEventSource.simulateError();
    });

    expect(screen.getByText(/reconnecting/i)).toBeInTheDocument();
  });

  it("closes connection on unmount", () => {
    const { unmount } = render(<LogViewer runId="test-123" />);
    unmount();

    expect(mockEventSource.close).toHaveBeenCalled();
  });
});
```

```typescript
// frontend/src/hooks/useSSE.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useSSE } from "./useSSE";

class MockEventSource {
  static instances: MockEventSource[] = [];
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  readyState = 1;

  constructor(public url: string) {
    MockEventSource.instances.push(this);
  }

  close = vi.fn();
}

describe("useSSE", () => {
  beforeEach(() => {
    MockEventSource.instances = [];
    vi.stubGlobal("EventSource", MockEventSource);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("connects to provided URL", () => {
    renderHook(() => useSSE("/api/test"));

    expect(MockEventSource.instances).toHaveLength(1);
    expect(MockEventSource.instances[0].url).toBe("/api/test");
  });

  it("returns messages as they arrive", async () => {
    const { result } = renderHook(() => useSSE<string>("/api/test"));

    act(() => {
      MockEventSource.instances[0].onmessage?.(
        new MessageEvent("message", { data: "test message" })
      );
    });

    await waitFor(() => {
      expect(result.current.data).toBe("test message");
    });
  });

  it("tracks connection state", async () => {
    const { result } = renderHook(() => useSSE("/api/test"));

    expect(result.current.isConnected).toBe(true);

    act(() => {
      MockEventSource.instances[0].readyState = 2;
      MockEventSource.instances[0].onerror?.(new Event("error"));
    });

    await waitFor(() => {
      expect(result.current.isConnected).toBe(false);
    });
  });

  it("reconnects on error with backoff", async () => {
    vi.useFakeTimers();

    renderHook(() => useSSE("/api/test", { reconnect: true }));

    // Simulate disconnect
    act(() => {
      MockEventSource.instances[0].onerror?.(new Event("error"));
    });

    // Fast-forward past reconnect delay
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    expect(MockEventSource.instances).toHaveLength(2);

    vi.useRealTimers();
  });

  it("closes connection on unmount", () => {
    const { unmount } = renderHook(() => useSSE("/api/test"));
    const instance = MockEventSource.instances[0];

    unmount();

    expect(instance.close).toHaveBeenCalled();
  });
});
```

---

## E2E Testing (Playwright)

### Configuration

```typescript
// frontend/playwright.config.ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,  // Max 1 retry; see Flake Policy
  workers: process.env.CI ? 1 : undefined,
  reporter: [["html", { open: "never" }]],

  use: {
    baseURL: "http://localhost:8765",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  webServer: {
    command: "cd .. && FUSION_GUI_FAKE_SIMULATOR=true fusion gui",
    url: "http://localhost:8765",
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});
```

### MVP E2E Tests (Required)

These two E2E tests are **required for MVP (M2)**:

```typescript
// frontend/e2e/create-run.spec.ts
import { test, expect } from "@playwright/test";

test.describe("Run Lifecycle", () => {
  test("user can create, monitor, and cancel a run", async ({ page }) => {
    // Navigate to home
    await page.goto("/");

    // Click "New Run" button
    await page.click('button:has-text("New Run")');

    // Fill in run details
    await page.fill('input[name="name"]', "E2E Test Run");
    await page.selectOption('select[name="template"]', "default");

    // Start the run
    await page.click('button:has-text("Start")');

    // Should redirect to run detail page
    await expect(page).toHaveURL(/\/runs\/[\w-]+/);

    // Wait for RUNNING status
    await expect(page.locator('[data-testid="run-status"]')).toHaveText(
      "RUNNING",
      { timeout: 5000 }
    );

    // Verify logs are streaming
    const logViewer = page.locator('[data-testid="log-viewer"]');
    await expect(logViewer).toBeVisible();

    // Wait for some log content
    await expect(logViewer).toContainText("Simulation started", {
      timeout: 10000,
    });

    // Cancel the run
    await page.click('button:has-text("Cancel")');

    // Confirm cancellation
    await page.click('button:has-text("Confirm")');

    // Verify cancelled status
    await expect(page.locator('[data-testid="run-status"]')).toHaveText(
      "CANCELLED",
      { timeout: 5000 }
    );
  });

  test("completed run shows correct status", async ({ page }) => {
    // Create a run that will complete quickly (fake simulator)
    await page.goto("/");
    await page.click('button:has-text("New Run")');
    await page.fill('input[name="name"]', "Quick Run");
    await page.selectOption('select[name="template"]', "quick");
    await page.click('button:has-text("Start")');

    // Wait for completion
    await expect(page.locator('[data-testid="run-status"]')).toHaveText(
      "COMPLETED",
      { timeout: 30000 }
    );

    // Verify final log message
    await expect(page.locator('[data-testid="log-viewer"]')).toContainText(
      "Simulation completed"
    );
  });
});
```

```typescript
// frontend/e2e/view-artifacts.spec.ts
import { test, expect } from "@playwright/test";
import { createCompletedRun } from "./helpers";

test.describe("Artifact Management", () => {
  test("user can browse and download artifacts from completed run", async ({
    page,
  }) => {
    // Create a completed run (helper uses API directly)
    const runId = await createCompletedRun(page);

    // Navigate to run detail
    await page.goto(`/runs/${runId}`);

    // Switch to artifacts tab
    await page.click('[data-testid="tab-artifacts"]');

    // Verify artifact list is visible
    const artifactList = page.locator('[data-testid="artifact-list"]');
    await expect(artifactList).toBeVisible();

    // Check for expected files
    await expect(artifactList).toContainText("output");
    await expect(artifactList).toContainText("sim.log");

    // Navigate into output directory
    await page.click('text="output"');

    // Should see result files
    await expect(artifactList).toContainText("_erlang.json");

    // Download a file
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.click('button:has-text("Download"):near(:text("10_erlang.json"))'),
    ]);

    // Verify download started
    expect(download.suggestedFilename()).toBe("10_erlang.json");
  });

  test("artifact browser shows file sizes", async ({ page }) => {
    const runId = await createCompletedRun(page);
    await page.goto(`/runs/${runId}`);
    await page.click('[data-testid="tab-artifacts"]');

    // Verify file sizes are displayed
    const fileRow = page.locator('[data-testid="artifact-row"]:has-text("sim.log")');
    await expect(fileRow.locator('[data-testid="file-size"]')).toBeVisible();
  });
});
```

### E2E Test Helpers

```typescript
// frontend/e2e/helpers.ts
import { Page, expect } from "@playwright/test";

/**
 * Create a completed run via API and return its ID.
 * Uses the fake simulator for fast completion.
 */
export async function createCompletedRun(page: Page): Promise<string> {
  const response = await page.request.post("/api/runs", {
    data: {
      name: "E2E Completed Run",
      template: "quick",
      config: {},
    },
  });

  expect(response.ok()).toBeTruthy();
  const { id } = await response.json();

  // Wait for run to complete (poll status)
  let status = "PENDING";
  let attempts = 0;
  while (status !== "COMPLETED" && attempts < 60) {
    await page.waitForTimeout(500);
    const statusResponse = await page.request.get(`/api/runs/${id}`);
    const data = await statusResponse.json();
    status = data.status;
    attempts++;

    if (status === "FAILED") {
      throw new Error(`Run ${id} failed unexpectedly`);
    }
  }

  if (status !== "COMPLETED") {
    throw new Error(`Run ${id} did not complete in time`);
  }

  return id;
}
```

---

## Fake Simulator Mode

The fake simulator enables fast, deterministic testing without running real simulations.

### Activation

Set environment variable before starting the server:

```bash
FUSION_GUI_FAKE_SIMULATOR=true fusion gui
```

Or in code:

```python
# fusion/api/config.py
import os

class Settings:
    FAKE_SIMULATOR = os.environ.get("FUSION_GUI_FAKE_SIMULATOR", "").lower() == "true"
```

### Behavior

| Aspect | Real Simulator | Fake Simulator |
|--------|---------------|----------------|
| Duration | Minutes to hours | 0.1-1 seconds |
| Output files | Full simulation data | Minimal stub data |
| Progress events | Real metrics | Synthetic values |
| Determinism | Varies | Fully deterministic |
| Log output | Detailed | Minimal markers |

### Fake Simulator Invocation

```python
# fusion/api/services/run_manager.py
from fusion.api.config import settings

def get_simulator_command(run_id: str, config_path: str) -> list[str]:
    """Get the command to run the simulator."""
    if settings.FAKE_SIMULATOR:
        return [
            sys.executable,
            str(Path(__file__).parent.parent / "devtools" / "fake_simulator.py"),
            "--config", config_path,
            "--output-dir", f"data/gui_runs/{run_id}/output",
            "--progress-file", f"data/gui_runs/{run_id}/progress.jsonl",
        ]
    else:
        return [
            sys.executable, "-m", "fusion.cli.run_sim",
            "--config", config_path,
            "--output-dir", f"data/gui_runs/{run_id}/output",
        ]
```

### Controlling Fake Simulator Behavior

Additional environment variables for testing edge cases:

| Variable | Description | Default |
|----------|-------------|---------|
| `FAKE_SIM_DURATION` | Total runtime in seconds | `0.1` |
| `FAKE_SIM_EXIT_CODE` | Exit code (0=success, 1=failure) | `0` |
| `FAKE_SIM_FAIL_AFTER` | Fail after N iterations | `-1` (never) |

Example: Testing failure handling:

```bash
FUSION_GUI_FAKE_SIMULATOR=true \
FAKE_SIM_EXIT_CODE=1 \
pytest fusion/api/tests/test_run_failure.py
```

---

## Test Commands

### Backend

```bash
# Run all backend tests
pytest fusion/api/tests/ -v

# Run with coverage
pytest fusion/api/tests/ --cov=fusion/api --cov-report=html

# Run only fast tests (exclude slow integration tests)
pytest fusion/api/tests/ -v -m "not slow"

# Run specific test file
pytest fusion/api/tests/test_runs.py -v
```

### Frontend

```bash
# Run all frontend tests
cd frontend && npm test

# Run in watch mode
cd frontend && npm run test:watch

# Run with coverage
cd frontend && npm run test:coverage

# Run specific test file
cd frontend && npm test -- RunCard.test.tsx
```

### E2E

```bash
# Run E2E tests (starts server automatically)
cd frontend && npm run test:e2e

# Run E2E tests with UI
cd frontend && npm run test:e2e:ui

# Run specific E2E test
cd frontend && npx playwright test create-run.spec.ts

# Debug E2E test
cd frontend && npx playwright test create-run.spec.ts --debug
```

### All Tests

```bash
# Run all tests (backend + frontend + e2e)
make test-gui

# CI mode (stricter, no watch)
make test-gui-ci
```

---

## CI Integration

See [08-ci-cd.md](08-ci-cd.md) for the full CI pipeline. Key test-related jobs:

1. **Backend Tests**: `pytest fusion/api/tests/ -v --tb=short`
2. **Frontend Tests**: `npm run test:ci`
3. **E2E Tests**: `npm run test:e2e` with `FUSION_GUI_FAKE_SIMULATOR=true`

All tests must pass before merging to `develop` or `main`.
