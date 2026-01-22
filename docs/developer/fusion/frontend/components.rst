.. _frontend-components:

==========
Components
==========

This page documents the reusable React components in ``frontend/src/components/``.

----

Layout Components
=================

:Location: ``src/components/layout/``

Components that define the overall page structure.

Layout.tsx
----------

Main layout wrapper that provides the header, sidebar, and content area.

.. code-block:: text

   import { Layout } from '@/components/layout/Layout'

   // Used in App.tsx as the root route element
   <Route path="/" element={<Layout />}>
     <Route index element={<RunListPage />} />
     {/* Child routes render in the Outlet */}
   </Route>

Structure:

.. code-block:: text

   ┌─────────────────────────────────────────────────┐
   │                    Header                        │
   ├──────────┬──────────────────────────────────────┤
   │          │                                       │
   │ Sidebar  │              Content                  │
   │          │             (Outlet)                  │
   │          │                                       │
   └──────────┴──────────────────────────────────────┘

Header.tsx
----------

Top navigation bar with logo, title, and global actions.

.. code-block:: text

   <header className="h-14 border-b bg-background flex items-center px-4">
     <Logo />
     <span className="font-semibold">FUSION</span>
     <div className="ml-auto">
       {/* Theme toggle, user menu, etc. */}
     </div>
   </header>

Sidebar.tsx
-----------

Navigation sidebar with links to all pages.

.. code-block:: text

   const navItems = [
     { path: '/', label: 'Runs', icon: PlayIcon },
     { path: '/topology', label: 'Topology', icon: NetworkIcon },
     { path: '/config', label: 'Config', icon: FileIcon },
     { path: '/codebase', label: 'Codebase', icon: CodeIcon },
     { path: '/settings', label: 'Settings', icon: SettingsIcon },
   ]

----

Run Components
==============

:Location: ``src/components/runs/``

Components for displaying and managing simulation runs.

RunCard.tsx
-----------

Card component displaying a single run in the list view.

.. code-block:: text

   interface RunCardProps {
     run: Run
     onClick?: () => void
   }

   function RunCard({ run, onClick }: RunCardProps) {
     return (
       <div className="border rounded-lg p-4 cursor-pointer hover:border-primary">
         <div className="flex items-center justify-between">
           <span className="font-medium">{run.name || run.id}</span>
           <RunStatusBadge status={run.status} />
         </div>
         <div className="text-sm text-muted-foreground">
           Created: {formatDate(run.created_at)}
         </div>
       </div>
     )
   }

RunStatusBadge.tsx
------------------

Status indicator badge with appropriate colors.

.. code-block:: text

   interface RunStatusBadgeProps {
     status: RunStatus
   }

   const statusColors = {
     PENDING: 'bg-yellow-100 text-yellow-800',
     RUNNING: 'bg-blue-100 text-blue-800',
     COMPLETED: 'bg-green-100 text-green-800',
     FAILED: 'bg-red-100 text-red-800',
     CANCELLED: 'bg-gray-100 text-gray-800',
   }

   function RunStatusBadge({ status }: RunStatusBadgeProps) {
     return (
       <span className={`px-2 py-1 rounded-full text-xs ${statusColors[status]}`}>
         {status}
       </span>
     )
   }

ProgressChart.tsx
-----------------

Visual progress indicator for running simulations.

.. code-block:: text

   interface ProgressChartProps {
     progress: RunProgress | null
   }

   function ProgressChart({ progress }: ProgressChartProps) {
     if (!progress) return null

     const percent = progress.percent_complete ?? 0

     return (
       <div className="space-y-2">
         <div className="flex justify-between text-sm">
           <span>Erlang {progress.current_erlang}</span>
           <span>{percent.toFixed(1)}%</span>
         </div>
         <div className="h-2 bg-gray-200 rounded">
           <div
             className="h-full bg-primary rounded"
             style={{ width: `${percent}%` }}
           />
         </div>
       </div>
     )
   }

----

Topology Components
===================

:Location: ``src/components/topology/``

Components for network visualization.

NetworkGraph.tsx
----------------

Interactive network graph using D3.js force-directed layout.

.. code-block:: text

   interface NetworkGraphProps {
     topology: TopologyResponse
     onNodeClick?: (nodeId: number) => void
     selectedNode?: number | null
   }

   function NetworkGraph({ topology, onNodeClick, selectedNode }: NetworkGraphProps) {
     const svgRef = useRef<SVGSVGElement>(null)

     useEffect(() => {
       // D3 force simulation setup
       const simulation = d3.forceSimulation(topology.nodes)
         .force('link', d3.forceLink(topology.links))
         .force('charge', d3.forceManyBody())
         .force('center', d3.forceCenter(width / 2, height / 2))

       // Draw nodes and links...
     }, [topology])

     return <svg ref={svgRef} className="w-full h-full" />
   }

Features:

- Force-directed layout with draggable nodes
- Click to select nodes
- Zoom and pan controls
- Link highlighting for selected node
- Optional utilization coloring

UtilizationLegend.tsx
---------------------

Color legend for link utilization visualization.

.. code-block:: text

   function UtilizationLegend() {
     return (
       <div className="flex items-center gap-2">
         <span className="text-sm">Utilization:</span>
         <div className="flex">
           <div className="w-4 h-4 bg-green-500" title="0-25%" />
           <div className="w-4 h-4 bg-yellow-500" title="25-50%" />
           <div className="w-4 h-4 bg-orange-500" title="50-75%" />
           <div className="w-4 h-4 bg-red-500" title="75-100%" />
         </div>
       </div>
     )
   }

----

Artifact Components
===================

:Location: ``src/components/artifacts/``

Components for browsing simulation output files.

ArtifactBrowser.tsx
-------------------

File browser for simulation artifacts with folder navigation.

.. code-block:: text

   interface ArtifactBrowserProps {
     runId: string
   }

   function ArtifactBrowser({ runId }: ArtifactBrowserProps) {
     const [currentPath, setCurrentPath] = useState('')

     const { data: artifacts } = useQuery({
       queryKey: ['artifacts', runId, currentPath],
       queryFn: () => artifactsApi.list(runId, currentPath),
     })

     return (
       <div>
         <Breadcrumb path={currentPath} onNavigate={setCurrentPath} />
         <FileList
           items={artifacts?.items ?? []}
           onFolderClick={(name) => setCurrentPath(`${currentPath}/${name}`)}
           onFileClick={(item) => window.open(item.downloadUrl)}
         />
       </div>
     )
   }

----

UI Components
=============

:Location: ``src/components/ui/``

Generic reusable UI components (buttons, inputs, cards, etc.).

These typically follow the shadcn/ui pattern - unstyled, composable components
with Tailwind classes.

Common Components
-----------------

- **Button** - Primary, secondary, outline variants
- **Input** - Text input with label and error state
- **Select** - Dropdown selection
- **Card** - Container with border and shadow
- **Badge** - Small status indicators
- **Tabs** - Tab navigation component
- **Dialog** - Modal dialog
- **Toast** - Notification messages

Example Usage
-------------

.. code-block:: text

   import { Button } from '@/components/ui/button'
   import { Input } from '@/components/ui/input'
   import { Card, CardHeader, CardContent } from '@/components/ui/card'

   function MyComponent() {
     return (
       <Card>
         <CardHeader>
           <h2>Form Title</h2>
         </CardHeader>
         <CardContent>
           <Input placeholder="Enter name" />
           <Button variant="primary">Submit</Button>
         </CardContent>
       </Card>
     )
   }
