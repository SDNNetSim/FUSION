# API Specification

Base URL: `http://localhost:8765/api`

All endpoints return JSON. Errors use standard HTTP status codes with a JSON body:

```json
{
  "detail": "Human-readable error message"
}
```

## Runs

### Create Run

```
POST /api/runs
```

**Request Body:**
```json
{
  "name": "My Simulation",
  "template": "default",
  "config": {
    "general_settings": {
      "network": "nsfnet",
      "erlang_start": 50,
      "erlang_stop": 200,
      "erlang_step": 10,
      "max_iters": 100
    },
    "topology_settings": {
      "k_paths": 3
    }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | No | User-friendly name (default: auto-generated) |
| `template` | string | No | Base template to use (default: "default") |
| `config` | object | Yes | Config overrides to apply on top of template |

**Response (201 Created):**
```json
{
  "id": "a1b2c3d4e5f6",
  "name": "My Simulation",
  "status": "PENDING",
  "created_at": "2024-01-15T10:00:00Z"
}
```

### List Runs

```
GET /api/runs
GET /api/runs?status=RUNNING
GET /api/runs?limit=10&offset=0
```

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `status` | string | Filter by status (comma-separated for multiple) |
| `limit` | int | Max results (default: 50, max: 100) |
| `offset` | int | Pagination offset (default: 0) |

**Response (200 OK):**
```json
{
  "runs": [
    {
      "id": "a1b2c3d4e5f6",
      "name": "My Simulation",
      "status": "RUNNING",
      "created_at": "2024-01-15T10:00:00Z",
      "started_at": "2024-01-15T10:00:01Z",
      "progress": {
        "current_erlang": 100,
        "total_erlangs": 16,
        "current_iteration": 45,
        "total_iterations": 100,
        "percent_complete": 35.2
      }
    }
  ],
  "total": 42,
  "limit": 50,
  "offset": 0
}
```

### Get Run

```
GET /api/runs/{run_id}
```

**Response (200 OK):**
```text
{
  "id": "a1b2c3d4e5f6",
  "name": "My Simulation",
  "status": "RUNNING",
  "config": { ... },
  "created_at": "2024-01-15T10:00:00Z",
  "started_at": "2024-01-15T10:00:01Z",
  "completed_at": null,
  "error_message": null,
  "progress": {
    "current_erlang": 100,
    "total_erlangs": 16,
    "current_iteration": 45,
    "total_iterations": 100,
    "percent_complete": 35.2,
    "latest_metrics": {
      "blocking_prob": 0.0234
    }
  }
}
```

### Cancel Run

```
DELETE /api/runs/{run_id}
```

Cancels a PENDING or RUNNING run. For COMPLETED/FAILED/CANCELLED runs, this deletes the run and its artifacts.

**Response (200 OK):**
```json
{
  "id": "a1b2c3d4e5f6",
  "status": "CANCELLED"
}
```

**Response (409 Conflict):** If run cannot be cancelled (e.g., already completed).

### Stream Logs (SSE)

```
GET /api/runs/{run_id}/logs
```

Server-Sent Events stream of log content. Supports reconnection and catch-up.

**Example (resuming from byte offset 4523):**
```
GET /api/runs/abc123/logs?offset=4523
```

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `from_start` | bool | Include existing log content (default: true) |
| `offset` | int | Byte offset to resume from (for reconnection) |

**Events:**
```
event: log
data: 2024-01-15 10:00:00 INFO Starting simulation...

event: log
data: 2024-01-15 10:00:01 INFO Loaded topology: nsfnet

event: heartbeat
data: {"offset": 4523}

event: end
data: COMPLETED
```

**Reconnection Protocol:**
1. Server sends `heartbeat` events every 15 seconds with current byte offset
2. Client stores the last received offset
3. On reconnect, client passes `?offset=<last_offset>` to resume without gaps
4. If `offset` is stale (log rotated), server returns 410 Gone; client should reconnect with `from_start=true`

**Client Behavior:**
- Use native `EventSource` which auto-reconnects on error
- Store last heartbeat offset in component state
- On reconnect, append `?offset=X` to URL

### Stream Progress (SSE)

```
GET /api/runs/{run_id}/progress
```

Server-Sent Events stream of progress updates. Supports reconnection and catch-up.

**Example (resuming from cursor):**
```
GET /api/runs/abc123/progress?cursor=evt_00045
```

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `from_start` | bool | Include all progress events (default: true) |
| `cursor` | string | Opaque cursor for resumption (from last event) |

**Events:**
```
event: progress
data: {"type":"iteration","erlang":50,"iteration":45,"total_iterations":100,"metrics":{"blocking_prob":0.023},"cursor":"evt_00045"}

event: progress
data: {"type":"erlang_complete","erlang":50,"metrics":{"mean_blocking":0.022},"cursor":"evt_00046"}

event: heartbeat
data: {"cursor":"evt_00046"}

event: end
data: COMPLETED
```

**Reconnection Protocol:**
1. Each progress event includes a `cursor` field
2. Server sends `heartbeat` every 15 seconds with current cursor
3. On reconnect, pass `?cursor=<last_cursor>` to resume
4. Server replays events after the cursor

## Artifacts

### List Artifacts

```
GET /api/runs/{run_id}/artifacts
GET /api/runs/{run_id}/artifacts?path=output
```

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `path` | string | Subdirectory to list (default: root) |

**Response (200 OK):**
```json
{
  "path": "output",
  "entries": [
    {
      "name": "sim_20240115_100000",
      "type": "directory",
      "modified_at": "2024-01-15T10:10:00Z"
    },
    {
      "name": "50_erlang.json",
      "type": "file",
      "size_bytes": 45678,
      "modified_at": "2024-01-15T10:05:00Z"
    }
  ]
}
```

### Download Artifact

```
GET /api/runs/{run_id}/artifacts/{path}
```

Downloads a file. Security checks are enforced:

**Security:**
- Path traversal blocked (403 Forbidden for `../` or absolute paths)
- Symlinks are resolved via `realpath()` and must resolve within the run directory
- Symlinks pointing outside run directory are rejected (403 Forbidden)

**Response Headers:**
```
Content-Type: application/octet-stream (or appropriate MIME type)
Content-Disposition: attachment; filename="50_erlang.json"
Content-Length: 45678
```

**Error Responses:**
- `403 Forbidden`: Path traversal attempt or symlink escape
- `404 Not Found`: File does not exist

### Preview Artifact

```
GET /api/runs/{run_id}/artifacts/{path}/preview
```

Returns file content for preview (text, JSON, CSV, images).

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `max_lines` | int | Limit lines for text files (default: 1000) |
| `max_size` | int | Limit bytes (default: 1MB) |

**Response (200 OK):**
```text
{
  "path": "output/sim/0/50_erlang.json",
  "type": "json",
  "content": { ... },
  "truncated": false
}
```

For images:
```json
{
  "path": "plots/blocking.png",
  "type": "image",
  "content_url": "/api/runs/{run_id}/artifacts/plots/blocking.png",
  "width": 800,
  "height": 600
}
```

## Configs

### List Templates

```
GET /api/configs/templates
```

**Response (200 OK):**
```json
{
  "templates": [
    {
      "name": "default",
      "description": "Full-featured production baseline",
      "path": "fusion/configs/templates/default.ini"
    },
    {
      "name": "minimal",
      "description": "Quick testing configuration",
      "path": "fusion/configs/templates/minimal.ini"
    }
  ]
}
```

### Get Template

```
GET /api/configs/templates/{name}
```

**Response (200 OK):**
```json
{
  "name": "default",
  "content": "[general_settings]\nnetwork = nsfnet\n...",
  "parsed": {
    "general_settings": {
      "network": "nsfnet",
      "erlang_start": 50
    }
  }
}
```

### Validate Config

```
POST /api/configs/validate
```

**Request Body:**
```json
{
  "config": {
    "general_settings": {
      "network": "invalid_network",
      "erlang_start": -10
    }
  }
}
```

**Response (200 OK - Valid):**
```json
{
  "valid": true,
  "warnings": []
}
```

**Response (200 OK - Invalid):**
```json
{
  "valid": false,
  "errors": [
    {
      "path": "general_settings.network",
      "message": "Unknown network: 'invalid_network'. Available: nsfnet, usnet, ..."
    },
    {
      "path": "general_settings.erlang_start",
      "message": "Value must be positive"
    }
  ],
  "warnings": [
    {
      "path": "general_settings.max_iters",
      "message": "Using default value: 100"
    }
  ]
}
```

### Get Config Schema

```
GET /api/configs/schema
```

Returns JSON Schema for configuration, enabling dynamic form generation.

**Response (200 OK):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "general_settings": {
      "type": "object",
      "properties": {
        "network": {
          "type": "string",
          "enum": ["nsfnet", "usnet", "cost239"],
          "description": "Network topology to simulate"
        },
        "erlang_start": {
          "type": "number",
          "minimum": 0,
          "description": "Starting Erlang value"
        }
      }
    }
  }
}
```

## Topology

### List Topologies

```
GET /api/topology
```

**Response (200 OK):**
```json
{
  "topologies": [
    {
      "name": "nsfnet",
      "nodes": 14,
      "links": 21,
      "description": "NSF Network topology"
    }
  ]
}
```

### Get Topology

```
GET /api/topology/{name}
```

**Response (200 OK):**
```json
{
  "name": "nsfnet",
  "nodes": [
    {"id": "0", "label": "Seattle", "x": 100, "y": 50},
    {"id": "1", "label": "San Francisco", "x": 80, "y": 150}
  ],
  "links": [
    {"source": "0", "target": "1", "weight": 1100, "slots": 320}
  ]
}
```

### Get Run Topology State

```
GET /api/runs/{run_id}/topology
```

Returns topology with current utilization from a running or completed simulation.

**Response (200 OK):**
```text
{
  "name": "nsfnet",
  "nodes": [...],
  "links": [
    {
      "source": "0",
      "target": "1",
      "slots_total": 320,
      "slots_used": 156,
      "utilization": 0.4875
    }
  ]
}
```

## System

### Health Check

```
GET /api/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "database": "connected",
  "active_runs": 1
}
```

### Version

```
GET /api/version
```

**Response (200 OK):**
```json
{
  "version": "6.1.0",
  "api_version": "1",
  "python_version": "3.11.5"
}
```

## Error Responses

| Status | Meaning |
|--------|---------|
| 400 | Bad Request - Invalid input |
| 403 | Forbidden - Path traversal or access denied |
| 404 | Not Found - Resource doesn't exist |
| 409 | Conflict - Operation not allowed in current state |
| 500 | Internal Server Error |

**Error Body:**
```json
{
  "detail": "Run not found: xyz123"
}
```

## Rate Limiting

No rate limiting for local use. If exposed over network (not recommended), consider adding.

## CORS

Not needed for production (same origin). In development, Vite proxy handles it.
