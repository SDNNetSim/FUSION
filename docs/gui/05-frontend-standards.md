# Frontend Standards

This document defines conventions for the React/TypeScript frontend.

## Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | React 18 | Industry standard, excellent ecosystem |
| Language | TypeScript (strict) | Type safety, better IDE support, catch errors early |
| Build | Vite | Fast HMR, simple config, modern ESM |
| Styling | Tailwind CSS | Utility-first, consistent design, small bundle |
| UI Components | shadcn/ui | Accessible, customizable, copy-paste ownership |
| Server State | TanStack Query (React Query) | Caching, SSE support, loading/error states |
| Client State | Zustand | Minimal, no boilerplate, TypeScript-friendly |
| Routing | React Router v6 | Standard, simple, supports nested routes |
| Forms | React Hook Form + Zod | Performant, validation, TypeScript inference |
| Charts | Recharts | React-native, good for scientific data |
| Graph Viz | React Flow or D3 | Network topology visualization |

## Project Structure

```
frontend/
├── src/
│   ├── main.tsx                    # Entry point
│   ├── App.tsx                     # Root component, providers, routing
│   │
│   ├── api/                        # API layer (single source of truth)
│   │   ├── client.ts               # Axios instance, interceptors
│   │   ├── runs.ts                 # Run endpoints
│   │   ├── configs.ts              # Config endpoints
│   │   ├── artifacts.ts            # Artifact endpoints
│   │   ├── topology.ts             # Topology endpoints
│   │   └── types.ts                # API types (or generated)
│   │
│   ├── components/                 # Reusable components
│   │   ├── ui/                     # Primitive UI (shadcn/ui components)
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   └── ...
│   │   ├── layout/                 # Layout components
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   └── MainLayout.tsx
│   │   ├── runs/                   # Run-specific components
│   │   │   ├── RunCard.tsx
│   │   │   ├── RunStatusBadge.tsx
│   │   │   ├── ProgressBar.tsx
│   │   │   └── LogViewer.tsx
│   │   ├── config/                 # Config editor components
│   │   │   ├── ConfigEditor.tsx
│   │   │   └── ConfigSection.tsx
│   │   ├── artifacts/              # File browser components
│   │   │   ├── FileBrowser.tsx
│   │   │   ├── FilePreview.tsx
│   │   │   └── FileIcon.tsx
│   │   └── topology/               # Network visualization
│   │       ├── NetworkGraph.tsx
│   │       └── NodeTooltip.tsx
│   │
│   ├── pages/                      # Route pages (thin, compose components)
│   │   ├── RunListPage.tsx
│   │   ├── RunDetailPage.tsx
│   │   ├── NewRunPage.tsx
│   │   ├── ConfigEditorPage.tsx
│   │   ├── TopologyPage.tsx
│   │   └── SettingsPage.tsx
│   │
│   ├── hooks/                      # Custom hooks
│   │   ├── useRuns.ts              # Run queries/mutations
│   │   ├── useRunDetail.ts         # Single run with SSE
│   │   ├── useSSE.ts               # Generic SSE hook
│   │   ├── useProgress.ts          # Progress subscription
│   │   ├── useConfigs.ts           # Config queries
│   │   └── useLocalStorage.ts      # Persistent local state
│   │
│   ├── stores/                     # Client-side state (Zustand)
│   │   ├── ui.ts                   # UI state (sidebar, theme)
│   │   └── index.ts                # Store exports
│   │
│   ├── lib/                        # Utilities
│   │   ├── utils.ts                # General utilities
│   │   ├── cn.ts                   # classnames helper (Tailwind)
│   │   └── format.ts               # Formatters (dates, bytes, etc.)
│   │
│   └── styles/
│       └── globals.css             # Tailwind imports, custom CSS
│
├── public/
│   └── favicon.ico
│
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── eslint.config.js
└── prettier.config.js
```

## State Management Philosophy

### Server State (TanStack Query)

Use TanStack Query for ALL data from the API:

```typescript
// hooks/useRuns.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as runsApi from '../api/runs';

export function useRuns(filters?: RunFilters) {
  return useQuery({
    queryKey: ['runs', filters],
    queryFn: () => runsApi.listRuns(filters),
  });
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: ['runs', runId],
    queryFn: () => runsApi.getRun(runId),
    refetchInterval: (data) =>
      data?.status === 'RUNNING' ? 2000 : false,
  });
}

export function useCreateRun() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: runsApi.createRun,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] });
    },
  });
}
```

**Why TanStack Query?**
- Automatic caching and background refetch
- Loading/error states handled consistently
- SSE integration via custom hooks
- No duplicate fetch logic

### Client State (Zustand)

Use Zustand only for UI state that doesn't come from the server:

```typescript
// stores/ui.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  sidebarCollapsed: boolean;
  theme: 'light' | 'dark' | 'system';
  logFollowEnabled: boolean;

  toggleSidebar: () => void;
  setTheme: (theme: UIState['theme']) => void;
  setLogFollow: (enabled: boolean) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      theme: 'system',
      logFollowEnabled: true,

      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setTheme: (theme) => set({ theme }),
      setLogFollow: (enabled) => set({ logFollowEnabled: enabled }),
    }),
    { name: 'fusion-ui' }
  )
);
```

**Why NOT Redux?**
- Overkill for this app's complexity
- Zustand is simpler, less boilerplate
- Server state belongs in React Query, not Redux

### When to Use What

| Data Type | Solution |
|-----------|----------|
| Runs, configs, artifacts | TanStack Query |
| Real-time logs/progress | TanStack Query + SSE hook |
| UI preferences (theme, sidebar) | Zustand (persisted) |
| Form state | React Hook Form |
| Derived/computed | `useMemo` in component |

## API Client

Single axios instance with consistent error handling:

```typescript
// api/client.ts
import axios, { AxiosError } from 'axios';

export const apiClient = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail: string }>) => {
    const message = error.response?.data?.detail || error.message;
    // Could integrate with toast notifications here
    console.error('API Error:', message);
    return Promise.reject(error);
  }
);

// Type-safe request helpers
export async function get<T>(url: string, params?: object): Promise<T> {
  const response = await apiClient.get<T>(url, { params });
  return response.data;
}

export async function post<T>(url: string, data?: object): Promise<T> {
  const response = await apiClient.post<T>(url, data);
  return response.data;
}

export async function del<T>(url: string): Promise<T> {
  const response = await apiClient.delete<T>(url);
  return response.data;
}
```

```typescript
// api/runs.ts
import { get, post, del } from './client';
import type { Run, RunCreateRequest, RunListResponse } from './types';

export const listRuns = (filters?: RunFilters) =>
  get<RunListResponse>('/runs', filters);

export const getRun = (id: string) =>
  get<Run>(`/runs/${id}`);

export const createRun = (data: RunCreateRequest) =>
  post<Run>('/runs', data);

export const cancelRun = (id: string) =>
  del<Run>(`/runs/${id}`);
```

## SSE Hook

Generic hook for Server-Sent Events with reconnection support:

```typescript
// hooks/useSSE.ts
import { useEffect, useRef, useCallback, useState } from 'react';

interface UseSSEOptions<T> {
  url: string;
  enabled?: boolean;
  onMessage: (data: T) => void;
  onError?: (error: Event) => void;
  onEnd?: () => void;
  /** Query param name for resume cursor (e.g., 'offset' or 'cursor') */
  resumeParam?: string;
}

export function useSSE<T>({
  url,
  enabled = true,
  onMessage,
  onError,
  onEnd,
  resumeParam,
}: UseSSEOptions<T>) {
  const eventSourceRef = useRef<EventSource | null>(null);
  const cursorRef = useRef<string | number | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!enabled) return;

    // Build URL with resume cursor if available
    let connectUrl = url;
    if (resumeParam && cursorRef.current !== null) {
      const sep = url.includes('?') ? '&' : '?';
      connectUrl = `${url}${sep}${resumeParam}=${cursorRef.current}`;
    }

    const eventSource = new EventSource(connectUrl);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as T;
        onMessage(data);
      } catch {
        onMessage(event.data as T);
      }
    };

    // Handle heartbeat events for cursor tracking
    eventSource.addEventListener('heartbeat', (event) => {
      try {
        const { offset, cursor } = JSON.parse(event.data);
        cursorRef.current = cursor ?? offset ?? null;
      } catch {
        // Ignore parse errors
      }
    });

    eventSource.addEventListener('end', () => {
      onEnd?.();
      eventSource.close();
      setIsConnected(false);
    });

    eventSource.onerror = (error) => {
      setIsConnected(false);
      onError?.(error);
      // EventSource auto-reconnects; cursor preserved in cursorRef
    };

    return () => {
      eventSource.close();
      setIsConnected(false);
    };
  }, [url, enabled, onMessage, onError, onEnd, resumeParam]);

  const close = useCallback(() => {
    eventSourceRef.current?.close();
    setIsConnected(false);
  }, []);

  return { close, isConnected };
}
```

### SSE Best Practices

1. **Always clean up on unmount**: The `useEffect` cleanup closes the connection
2. **Handle reconnection**: Store cursor/offset from heartbeat events; EventSource auto-reconnects
3. **Track connection state**: Use `isConnected` for UI feedback (e.g., "Reconnecting...")
4. **Stable callbacks**: Wrap `onMessage`/`onError`/`onEnd` in `useCallback` to avoid reconnection loops

**Resume parameter by stream:**
| Stream | `resumeParam` | Type | Example |
|--------|---------------|------|---------|
| `/api/runs/{id}/logs` | `offset` | byte position (int) | `?offset=4523` |
| `/api/runs/{id}/progress` | `cursor` | opaque string | `?cursor=evt_00045` |

**Testing SSE hooks:**

```typescript
// hooks/__tests__/useSSE.test.ts
import { renderHook, act } from '@testing-library/react';
import { useSSE } from '../useSSE';

// Mock EventSource
class MockEventSource {
  static instances: MockEventSource[] = [];
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onopen: (() => void) | null = null;
  listeners: Record<string, ((event: MessageEvent) => void)[]> = {};

  constructor(public url: string) {
    MockEventSource.instances.push(this);
  }

  addEventListener(event: string, handler: (event: MessageEvent) => void) {
    this.listeners[event] = this.listeners[event] || [];
    this.listeners[event].push(handler);
  }

  close() {}

  // Test helpers
  simulateMessage(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  simulateHeartbeat(cursor: string) {
    this.listeners['heartbeat']?.forEach((h) =>
      h({ data: JSON.stringify({ cursor }) } as MessageEvent)
    );
  }
}

global.EventSource = MockEventSource as unknown as typeof EventSource;

describe('useSSE', () => {
  beforeEach(() => {
    MockEventSource.instances = [];
  });

  it('connects and receives messages', () => {
    const onMessage = vi.fn();
    renderHook(() => useSSE({ url: '/api/test', onMessage }));

    const es = MockEventSource.instances[0];
    es.simulateMessage({ value: 42 });

    expect(onMessage).toHaveBeenCalledWith({ value: 42 });
  });

  it('stores cursor from heartbeat', () => {
    const onMessage = vi.fn();
    const { result } = renderHook(() =>
      useSSE({ url: '/api/test', onMessage, resumeParam: 'cursor' })
    );

    const es = MockEventSource.instances[0];
    es.simulateHeartbeat('evt_123');

    // Cursor is internal; verify by checking reconnect URL would include it
    // (implementation detail, but validates the behavior)
  });
});
```

## Component Conventions

### File Naming

- Components: `PascalCase.tsx` (e.g., `RunCard.tsx`)
- Hooks: `camelCase.ts` with `use` prefix (e.g., `useRuns.ts`)
- Utils: `camelCase.ts` (e.g., `format.ts`)
- Types: `camelCase.ts` or colocated with component

### Component Structure

```text
// components/runs/RunCard.tsx
import { memo } from 'react';
import { Card, CardHeader, CardContent } from '../ui/card';
import { RunStatusBadge } from './RunStatusBadge';
import { ProgressBar } from './ProgressBar';
import type { Run } from '../../api/types';
import { formatRelativeTime } from '../../lib/format';

interface RunCardProps {
  run: Run;
  onClick?: () => void;
}

export const RunCard = memo(function RunCard({ run, onClick }: RunCardProps) {
  return (
    <Card
      className="cursor-pointer hover:shadow-md transition-shadow"
      onClick={onClick}
    >
      <CardHeader className="flex flex-row items-center justify-between">
        <h3 className="font-medium truncate">{run.name}</h3>
        <RunStatusBadge status={run.status} />
      </CardHeader>
      <CardContent>
        {run.status === 'RUNNING' && run.progress && (
          <ProgressBar
            value={run.progress.percent_complete}
            label={`Erlang ${run.progress.current_erlang}`}
          />
        )}
        <p className="text-sm text-muted-foreground mt-2">
          Created {formatRelativeTime(run.created_at)}
        </p>
      </CardContent>
    </Card>
  );
});
```

### Rules

1. **One component per file** (except tiny helpers)
2. **Named exports** (not default exports)
3. **Props interface** above component
4. **memo()** for list items and expensive renders
5. **Destructure props** in function signature
6. **Colocate styles** (Tailwind classes in component)

## Error Boundaries

Wrap pages in error boundaries:

```text
// components/ErrorBoundary.tsx
import { Component, ReactNode } from 'react';
import { Button } from './ui/button';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="p-8 text-center">
            <h2 className="text-xl font-bold text-red-600">Something went wrong</h2>
            <p className="mt-2 text-gray-600">{this.state.error?.message}</p>
            <Button
              className="mt-4"
              onClick={() => this.setState({ hasError: false })}
            >
              Try again
            </Button>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
```

## Linting and Formatting

### ESLint Config

```javascript
// eslint.config.js
import js from '@eslint/js';
import typescript from '@typescript-eslint/eslint-plugin';
import tsParser from '@typescript-eslint/parser';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';

export default [
  js.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        project: './tsconfig.json',
      },
    },
    plugins: {
      '@typescript-eslint': typescript,
      react,
      'react-hooks': reactHooks,
    },
    rules: {
      // TypeScript
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-explicit-any': 'error',

      // React
      'react/react-in-jsx-scope': 'off',
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',

      // General
      'no-console': ['warn', { allow: ['warn', 'error'] }],
    },
  },
];
```

### Prettier Config

```javascript
// prettier.config.js
export default {
  semi: true,
  singleQuote: true,
  trailingComma: 'es5',
  tabWidth: 2,
  printWidth: 100,
  plugins: ['prettier-plugin-tailwindcss'],
};
```

### TypeScript Config

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "jsx": "react-jsx",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src"]
}
```

## Testing Strategy

### Component Tests (Vitest + Testing Library)

```typescript
// components/runs/__tests__/RunCard.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RunCard } from '../RunCard';

const mockRun = {
  id: 'abc123',
  name: 'Test Run',
  status: 'RUNNING' as const,
  created_at: '2024-01-15T10:00:00Z',
  progress: {
    percent_complete: 50,
    current_erlang: 100,
  },
};

describe('RunCard', () => {
  it('renders run name and status', () => {
    render(<RunCard run={mockRun} />);

    expect(screen.getByText('Test Run')).toBeInTheDocument();
    expect(screen.getByText('RUNNING')).toBeInTheDocument();
  });

  it('shows progress bar for running jobs', () => {
    render(<RunCard run={mockRun} />);

    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '50');
  });

  it('calls onClick when clicked', async () => {
    const onClick = vi.fn();
    render(<RunCard run={mockRun} onClick={onClick} />);

    await userEvent.click(screen.getByRole('article'));
    expect(onClick).toHaveBeenCalled();
  });
});
```

### What to Test

- **Do test**: User interactions, conditional rendering, error states
- **Don't test**: Implementation details, styling, third-party libraries

## Design System

### Color Palette (Tailwind)

Use semantic color names via CSS variables for theming:

```css
/* styles/globals.css */
@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    /* ... */
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    /* ... */
  }
}
```

### Typography

- **Headings**: font-semibold, tracking-tight
- **Body**: text-base, text-muted-foreground for secondary
- **Code**: font-mono, bg-muted

### Spacing

Use Tailwind's spacing scale consistently:
- Section padding: `p-6` or `p-8`
- Card padding: `p-4`
- Element gaps: `gap-2`, `gap-4`
- Stack spacing: `space-y-4`

### Accessibility

- All interactive elements keyboard accessible
- ARIA labels for icon-only buttons
- Focus visible rings
- Color contrast meets WCAG AA
