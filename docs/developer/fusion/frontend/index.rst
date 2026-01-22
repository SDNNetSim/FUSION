.. _frontend-module:

===============
Frontend Module
===============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Web-based user interface for the FUSION GUI
   :Location: ``frontend/``
   :Build Output: ``fusion/api/static/``
   :Tech Stack: React 18, TypeScript, Vite, TailwindCSS
   :Key Libraries: React Query, React Router, Monaco Editor, D3.js

The frontend module contains the React application that provides the web-based
GUI for FUSION. It communicates with the FastAPI backend via REST APIs and
Server-Sent Events.

Developers work here when adding new UI features, modifying pages, or updating
the visual design.

Key Concepts
============

React Query (TanStack Query)
----------------------------

Data fetching and caching is handled by React Query:

.. code-block:: text

   import { useQuery } from '@tanstack/react-query'
   import { runsApi } from '@/api/client'

   function RunListPage() {
     const { data, isLoading, error } = useQuery({
       queryKey: ['runs'],
       queryFn: () => runsApi.list(),
       refetchInterval: 5000,  // Poll every 5 seconds
     })
     // ...
   }

Benefits:

- Automatic caching and refetching
- Loading and error states
- Optimistic updates for mutations
- Polling for real-time updates

Monaco Editor
-------------

The configuration editor uses Monaco Editor (the editor from VS Code):

.. code-block:: text

   import Editor from '@monaco-editor/react'

   <Editor
     language="ini"
     value={configContent}
     onChange={setConfigContent}
     theme={isDark ? 'vs-dark' : 'light'}
   />

Server-Sent Events (SSE)
------------------------

Log and progress streaming use the EventSource API:

.. code-block:: text

   const eventSource = new EventSource(`/api/runs/${runId}/logs`)

   eventSource.addEventListener('log', (event) => {
     const data = JSON.parse(event.data)
     appendLog(data.line)
   })

   eventSource.addEventListener('done', () => {
     eventSource.close()
   })

TailwindCSS
-----------

Styling uses TailwindCSS utility classes:

.. code-block:: text

   <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
     Start Run
   </button>

Architecture
============

Module Structure
----------------

.. code-block:: text

   frontend/
   ├── src/
   │   ├── api/                  # API client and types
   │   │   ├── client.ts         # Axios-based API functions
   │   │   └── types.ts          # TypeScript interfaces
   │   ├── components/           # Reusable React components
   │   │   ├── layout/           # Header, Sidebar, Layout
   │   │   ├── runs/             # Run-related components
   │   │   ├── topology/         # Network visualization
   │   │   ├── artifacts/        # File browser components
   │   │   └── ui/               # Generic UI components
   │   ├── pages/                # Page components (routes)
   │   │   ├── RunListPage.tsx   # Dashboard / run list
   │   │   ├── NewRunPage.tsx    # Create new run form
   │   │   ├── RunDetailPage.tsx # Run details with tabs
   │   │   ├── TopologyPage.tsx  # Network visualization
   │   │   ├── ConfigEditorPage.tsx  # INI editor
   │   │   ├── CodebaseExplorerPage.tsx  # Code browser
   │   │   └── SettingsPage.tsx  # User preferences
   │   ├── hooks/                # Custom React hooks
   │   ├── stores/               # State management (Zustand)
   │   ├── styles/               # Global CSS styles
   │   ├── lib/                  # Utility functions
   │   ├── App.tsx               # Main app with routing
   │   └── main.tsx              # Entry point
   ├── public/                   # Static assets
   ├── package.json              # Dependencies
   ├── vite.config.ts            # Vite configuration
   ├── tailwind.config.js        # Tailwind configuration
   └── tsconfig.json             # TypeScript configuration

Page Overview
=============

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Page
     - Route
     - Description
   * - RunListPage
     - ``/`` or ``/runs``
     - Dashboard showing all simulation runs
   * - NewRunPage
     - ``/runs/new``
     - Form to create a new simulation run
   * - RunDetailPage
     - ``/runs/:runId``
     - Run details with logs and artifacts tabs
   * - TopologyPage
     - ``/topology``
     - Interactive network topology viewer
   * - ConfigEditorPage
     - ``/config``
     - INI configuration file editor
   * - CodebaseExplorerPage
     - ``/codebase``
     - Codebase browser with architecture view
   * - SettingsPage
     - ``/settings``
     - User preferences (theme, etc.)

Development Guide
=================

Prerequisites
-------------

- Node.js 18+ (LTS recommended)
- npm 9+

Getting Started
---------------

.. code-block:: bash

   cd frontend

   # Install dependencies
   npm install

   # Start development server (with hot reload)
   npm run dev

   # The dev server runs at http://localhost:5173
   # API requests are proxied to the backend at http://localhost:8765

Building for Production
-----------------------

.. code-block:: bash

   # Build optimized bundle
   npm run build

   # Output is placed in fusion/api/static/
   # The backend serves these files directly

Code Quality
------------

.. code-block:: bash

   # Run linter
   npm run lint

   # Type check
   npm run type-check

   # Format code (Prettier)
   npm run format

Adding a New Page
-----------------

**1. Create the page component** in ``src/pages/``:

.. code-block:: text

   // src/pages/MyNewPage.tsx
   export function MyNewPage() {
     return (
       <div className="p-6">
         <h1 className="text-2xl font-bold">My New Page</h1>
         {/* ... */}
       </div>
     )
   }

**2. Add the route** in ``src/App.tsx``:

.. code-block:: text

   import { MyNewPage } from '@/pages/MyNewPage'

   // Inside Routes
   <Route path="my-page" element={<MyNewPage />} />

**3. Add navigation** in ``src/components/layout/Sidebar.tsx``.

Adding a New API Endpoint
-------------------------

**1. Add types** in ``src/api/types.ts``:

.. code-block:: text

   export interface MyData {
     id: string
     name: string
   }

**2. Add API function** in ``src/api/client.ts``:

.. code-block:: text

   export const myApi = {
     getData: async () => {
       const { data } = await api.get<MyData[]>('/my-endpoint')
       return data
     },
   }

**3. Use with React Query** in your component:

.. code-block:: text

   const { data } = useQuery({
     queryKey: ['my-data'],
     queryFn: myApi.getData,
   })

Environment Variables
---------------------

Create ``.env.local`` for local development:

.. code-block:: text

   VITE_API_BASE_URL=http://localhost:8765

----

Contents
========

.. toctree::
   :maxdepth: 1

   components
   pages
   api-client
