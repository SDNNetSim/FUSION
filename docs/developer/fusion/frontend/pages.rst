.. _frontend-pages:

=====
Pages
=====

This page documents the page components in ``frontend/src/pages/``. Each page
corresponds to a route in the application.

----

RunListPage
===========

:Location: ``src/pages/RunListPage.tsx``
:Route: ``/`` or ``/runs``

The main dashboard showing all simulation runs.

Features
--------

- Lists all runs with status indicators
- Automatic refresh (polls every 5 seconds)
- Filter by status (completed, running, failed)
- Click to navigate to run details
- "New Run" button to create a simulation

Implementation
--------------

.. code-block:: text

   function RunListPage() {
     const navigate = useNavigate()

     const { data, isLoading } = useQuery({
       queryKey: ['runs'],
       queryFn: () => runsApi.list(),
       refetchInterval: 5000,
     })

     return (
       <div className="p-6">
         <div className="flex justify-between items-center mb-6">
           <h1 className="text-2xl font-bold">Simulation Runs</h1>
           <Button onClick={() => navigate('/runs/new')}>New Run</Button>
         </div>

         {isLoading ? (
           <LoadingSpinner />
         ) : (
           <div className="grid gap-4">
             {data?.runs.map((run) => (
               <RunCard
                 key={run.id}
                 run={run}
                 onClick={() => navigate(`/runs/${run.id}`)}
               />
             ))}
           </div>
         )}
       </div>
     )
   }

----

NewRunPage
==========

:Location: ``src/pages/NewRunPage.tsx``
:Route: ``/runs/new``

Form for creating a new simulation run.

Features
--------

- Run ID input (auto-generated if empty)
- Optional display name
- Template selection dropdown
- Configuration editor (Monaco) for customization
- Validation before submission

Implementation
--------------

.. code-block:: text

   function NewRunPage() {
     const navigate = useNavigate()
     const [runId, setRunId] = useState('')
     const [template, setTemplate] = useState('minimal.ini')
     const [configContent, setConfigContent] = useState('')

     const { data: templates } = useQuery({
       queryKey: ['templates'],
       queryFn: configsApi.listTemplates,
     })

     const createMutation = useMutation({
       mutationFn: runsApi.create,
       onSuccess: (run) => navigate(`/runs/${run.id}`),
     })

     const handleSubmit = () => {
       createMutation.mutate({
         run_id: runId || undefined,
         template,
         config_content: configContent,
       })
     }

     return (
       <div className="p-6 max-w-3xl">
         <h1 className="text-2xl font-bold mb-6">New Simulation Run</h1>

         <form onSubmit={handleSubmit}>
           <Input label="Run ID" value={runId} onChange={setRunId} />
           <Select label="Template" options={templates} value={template} onChange={setTemplate} />
           <Editor value={configContent} onChange={setConfigContent} language="ini" />
           <Button type="submit" loading={createMutation.isLoading}>
             Start Run
           </Button>
         </form>
       </div>
     )
   }

----

RunDetailPage
=============

:Location: ``src/pages/RunDetailPage.tsx``
:Route: ``/runs/:runId``

Detailed view of a single run with tabs for different information.

Features
--------

- Run metadata (status, timestamps, template)
- Progress indicator for running simulations
- Logs tab with real-time streaming
- Artifacts tab with file browser
- Cancel button for running simulations

Tabs
----

**Logs Tab**

- Displays simulation output in a scrollable terminal-like view
- Uses Server-Sent Events for real-time updates
- Auto-scrolls to bottom for new content
- Preserves scroll position when manually scrolled up

**Artifacts Tab**

- File browser for simulation output files
- Navigate folders (breadcrumb navigation)
- Click files to download
- Preview JSON files inline

Implementation (Logs)
---------------------

.. code-block:: text

   function LogsPanel({ runId }: { runId: string }) {
     const [logs, setLogs] = useState<string[]>([])
     const logsEndRef = useRef<HTMLDivElement>(null)

     useEffect(() => {
       const eventSource = new EventSource(runsApi.getLogsUrl(runId))

       eventSource.addEventListener('log', (e) => {
         const data = JSON.parse(e.data)
         setLogs((prev) => [...prev, data.line])
       })

       eventSource.addEventListener('done', () => {
         eventSource.close()
       })

       return () => eventSource.close()
     }, [runId])

     // Auto-scroll effect
     useEffect(() => {
       logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
     }, [logs])

     return (
       <div className="font-mono text-sm bg-black text-green-400 p-4 h-96 overflow-auto">
         {logs.map((line, i) => (
           <div key={i}>{line}</div>
         ))}
         <div ref={logsEndRef} />
       </div>
     )
   }

----

TopologyPage
============

:Location: ``src/pages/TopologyPage.tsx``
:Route: ``/topology``

Interactive network topology visualization.

Features
--------

- Dropdown to select topology (NSFNet, USNet, etc.)
- Force-directed graph visualization
- Click nodes to see details
- Zoom and pan controls
- Node degree information on selection

Implementation
--------------

.. code-block:: text

   function TopologyPage() {
     const [selectedTopology, setSelectedTopology] = useState('NSFNet')
     const [selectedNode, setSelectedNode] = useState<number | null>(null)

     const { data: topologies } = useQuery({
       queryKey: ['topologies'],
       queryFn: topologyApi.list,
     })

     const { data: topology } = useQuery({
       queryKey: ['topology', selectedTopology],
       queryFn: () => topologyApi.get(selectedTopology),
       enabled: !!selectedTopology,
     })

     return (
       <div className="p-6 h-full flex flex-col">
         <div className="flex items-center gap-4 mb-4">
           <Select
             value={selectedTopology}
             onChange={setSelectedTopology}
             options={topologies?.topologies ?? []}
           />
           {selectedNode !== null && (
             <NodeInfo node={topology?.nodes[selectedNode]} />
           )}
         </div>

         <div className="flex-1 border rounded">
           {topology && (
             <NetworkGraph
               topology={topology}
               selectedNode={selectedNode}
               onNodeClick={setSelectedNode}
             />
           )}
         </div>
       </div>
     )
   }

----

ConfigEditorPage
================

:Location: ``src/pages/ConfigEditorPage.tsx``
:Route: ``/config``

INI configuration file editor.

Features
--------

- Template selection dropdown
- Monaco editor with INI syntax highlighting
- Dark/light theme support
- Copy to clipboard
- Use with new run (navigates to NewRunPage with config)

Implementation
--------------

.. code-block:: text

   function ConfigEditorPage() {
     const [template, setTemplate] = useState<string | null>(null)
     const [content, setContent] = useState('')
     const { theme } = useTheme()

     const { data: templates } = useQuery({
       queryKey: ['templates'],
       queryFn: configsApi.listTemplates,
     })

     const { data: templateData } = useQuery({
       queryKey: ['template', template],
       queryFn: () => configsApi.getTemplate(template!),
       enabled: !!template,
     })

     useEffect(() => {
       if (templateData) {
         setContent(templateData.content)
       }
     }, [templateData])

     return (
       <div className="p-6 h-full flex flex-col">
         <div className="flex items-center gap-4 mb-4">
           <Select
             label="Template"
             value={template}
             onChange={setTemplate}
             options={templates?.templates.map(t => t.name) ?? []}
           />
         </div>

         <div className="flex-1 border rounded overflow-hidden">
           <Editor
             value={content}
             onChange={(value) => setContent(value ?? '')}
             language="ini"
             theme={theme === 'dark' ? 'vs-dark' : 'light'}
           />
         </div>
       </div>
     )
   }

----

CodebaseExplorerPage
====================

:Location: ``src/pages/CodebaseExplorerPage.tsx``
:Route: ``/codebase``

Browse and explore the FUSION codebase.

Features
--------

- Architecture view with module cards
- Guided tour for newcomers
- File tree navigation
- Code viewer with syntax highlighting
- File search

Views
-----

**Architecture View**

Grid of cards showing high-level modules (core, modules, configs, etc.).
Each card shows the module name, description, and key files.

**Code View**

Split panel with file tree on left and code viewer on right.
Supports syntax highlighting for Python, INI, TypeScript, etc.

**Tour Mode**

Step-by-step walkthrough of the codebase structure with
highlighted sections and explanatory text.

Implementation
--------------

.. code-block:: text

   function CodebaseExplorerPage() {
     const [view, setView] = useState<'architecture' | 'code'>('architecture')
     const [selectedFile, setSelectedFile] = useState<string | null>(null)
     const [tourActive, setTourActive] = useState(false)

     const { data: tree } = useQuery({
       queryKey: ['codebase-tree'],
       queryFn: codebaseApi.getTree,
     })

     const { data: fileContent } = useQuery({
       queryKey: ['codebase-file', selectedFile],
       queryFn: () => codebaseApi.getFile(selectedFile!),
       enabled: !!selectedFile,
     })

     if (view === 'architecture') {
       return <ArchitectureView onSelectModule={...} onStartTour={...} />
     }

     return (
       <div className="h-full flex">
         <FileTree tree={tree} onSelect={setSelectedFile} />
         <CodeViewer content={fileContent?.content} language={...} />
       </div>
     )
   }

----

SettingsPage
============

:Location: ``src/pages/SettingsPage.tsx``
:Route: ``/settings``

User preferences and settings.

Features
--------

- Theme selection (light, dark, system)
- Settings persisted to localStorage

Implementation
--------------

.. code-block:: text

   function SettingsPage() {
     const { theme, setTheme } = useTheme()

     return (
       <div className="p-6 max-w-2xl">
         <h1 className="text-2xl font-bold mb-6">Settings</h1>

         <Card>
           <CardHeader>Appearance</CardHeader>
           <CardContent>
             <label className="block mb-2">Theme</label>
             <div className="flex gap-2">
               {['light', 'dark', 'system'].map((t) => (
                 <Button
                   key={t}
                   variant={theme === t ? 'primary' : 'outline'}
                   onClick={() => setTheme(t)}
                 >
                   {t.charAt(0).toUpperCase() + t.slice(1)}
                 </Button>
               ))}
             </div>
           </CardContent>
         </Card>
       </div>
     )
   }
