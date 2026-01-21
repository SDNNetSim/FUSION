# FUSION GUI Overview

This directory contains the design documentation for the FUSION web-based graphical user interface.

## What is FUSION GUI?

FUSION GUI is a browser-based interface for the FUSION optical network simulator. It allows users to:

- Configure and launch simulations without using the CLI
- Monitor running simulations with real-time progress and logs
- Browse and download output artifacts
- Visualize network topology and results
- Edit configuration files with validation

## Design Philosophy

1. **Local-first**: Runs on `localhost`, single-user, no authentication required
2. **Zero friction for users**: `pip install fusion[gui]` then `fusion gui` - no Node.js needed
3. **Developer friendly**: Hot reload for both frontend and backend during development
4. **Subprocess isolation**: Simulations run as separate processes for stability and clean cancellation
5. **Stateless where possible**: Server can restart without losing track of running jobs

## Documentation Index

| Document | Description |
|----------|-------------|
| [01-architecture.md](01-architecture.md) | System architecture and component overview |
| [02-run-lifecycle.md](02-run-lifecycle.md) | Simulation run states and transitions |
| [03-run-directory-contract.md](03-run-directory-contract.md) | File/directory structure for each run |
| [04-api.md](04-api.md) | REST API specification |
| [05-frontend-standards.md](05-frontend-standards.md) | React/TypeScript conventions and patterns |
| [06-backend-standards.md](06-backend-standards.md) | Python/FastAPI conventions and patterns |
| [07-contributing.md](07-contributing.md) | Contribution guidelines and PR checklist |
| [08-ci-cd.md](08-ci-cd.md) | CI/CD pipeline and release process |
| [09-roadmap.md](09-roadmap.md) | Phased implementation plan with milestones |
| [10-testing.md](10-testing.md) | Testing strategy and requirements |

## Diagrams

- [diagrams/architecture.txt](diagrams/architecture.txt) - System architecture
- [diagrams/run-state-machine.txt](diagrams/run-state-machine.txt) - Run lifecycle state machine

## Architecture Decision Records

- [adr/0001-react-fastapi-architecture.md](adr/0001-react-fastapi-architecture.md) - Why React + FastAPI

## Quick Links

- **For users**: See [09-roadmap.md](09-roadmap.md) for feature timeline
- **For frontend devs**: Start with [05-frontend-standards.md](05-frontend-standards.md)
- **For backend devs**: Start with [06-backend-standards.md](06-backend-standards.md)
- **For contributors**: Read [07-contributing.md](07-contributing.md) before your first PR
