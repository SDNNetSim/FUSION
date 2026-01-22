.. _frontend-api-client:

==========
API Client
==========

This page documents the API client layer in ``frontend/src/api/``.

----

client.ts - API Functions
=========================

:Location: ``src/api/client.ts``

Provides typed functions for all API endpoints using Axios.

Setup
-----

.. code-block:: text

   import axios from 'axios'

   const api = axios.create({
     baseURL: '/api',
     headers: {
       'Content-Type': 'application/json',
     },
   })

The base URL ``/api`` works with both:

- Development: Vite proxies requests to the backend
- Production: Same origin, backend serves the frontend

API Objects
-----------

runsApi
^^^^^^^

.. code-block:: text

   export const runsApi = {
     // List all runs with optional filtering
     list: async (params?: { status?: string; limit?: number; offset?: number }) => {
       const { data } = await api.get<RunListResponse>('/runs', { params })
       return data
     },

     // Get a specific run by ID
     get: async (runId: string) => {
       const { data } = await api.get<Run>(`/runs/${runId}`)
       return data
     },

     // Create a new run
     create: async (run: RunCreate) => {
       const { data } = await api.post<Run>('/runs', run)
       return data
     },

     // Cancel or delete a run
     cancel: async (runId: string) => {
       const { data } = await api.delete<Run>(`/runs/${runId}`)
       return data
     },

     // Get the SSE URL for log streaming
     getLogsUrl: (runId: string, fromStart = true) => {
       return `/api/runs/${runId}/logs?from_start=${fromStart}`
     },
   }

configsApi
^^^^^^^^^^

.. code-block:: text

   export const configsApi = {
     // List available configuration templates
     listTemplates: async () => {
       const { data } = await api.get<TemplateListResponse>('/configs/templates')
       return data
     },

     // Get content of a specific template
     getTemplate: async (name: string) => {
       const { data } = await api.get<TemplateContent>(`/configs/templates/${name}`)
       return data
     },
   }

artifactsApi
^^^^^^^^^^^^

.. code-block:: text

   export const artifactsApi = {
     // List artifacts for a run
     list: async (runId: string, path = '') => {
       const { data } = await api.get<ArtifactListResponse>(`/runs/${runId}/artifacts`, {
         params: path ? { path } : undefined,
       })
       return data
     },

     // Get download URL for an artifact
     getDownloadUrl: (runId: string, filePath: string) => {
       return `/api/runs/${runId}/artifacts/${filePath}`
     },
   }

topologyApi
^^^^^^^^^^^

.. code-block:: text

   export const topologyApi = {
     // List available topologies
     list: async () => {
       const { data } = await api.get<TopologyListResponse>('/topology')
       return data
     },

     // Get topology data with node positions
     get: async (name: string) => {
       const { data } = await api.get<TopologyResponse>(`/topology/${name}`)
       return data
     },
   }

codebaseApi
^^^^^^^^^^^

.. code-block:: text

   export const codebaseApi = {
     // Get directory tree
     getTree: async () => {
       const { data } = await api.get<ModuleTreeResponse>('/codebase/tree')
       return data
     },

     // Get file content
     getFile: async (path: string) => {
       const { data } = await api.get<FileContent>(`/codebase/file/${path}`)
       return data
     },

     // Search files by name
     search: async (query: string, limit = 20) => {
       const { data } = await api.get<SearchResult[]>('/codebase/search', {
         params: { q: query, limit },
       })
       return data
     },
   }

systemApi
^^^^^^^^^

.. code-block:: text

   export const systemApi = {
     // Health check
     health: async () => {
       const { data } = await api.get<HealthResponse>('/health')
       return data
     },
   }

----

types.ts - TypeScript Interfaces
================================

:Location: ``src/api/types.ts``

Defines TypeScript interfaces for all API request/response types.

Run Types
---------

.. code-block:: text

   // Run status values
   export type RunStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'

   // Progress information for running simulations
   export interface RunProgress {
     current_erlang: number | null
     total_erlangs: number | null
     current_iteration: number | null
     total_iterations: number | null
     percent_complete: number | null
   }

   // Full run object
   export interface Run {
     id: string
     name: string | null
     status: RunStatus
     template: string
     created_at: string
     started_at: string | null
     completed_at: string | null
     error_message: string | null
     progress: RunProgress | null
   }

   // Request body for creating a run
   export interface RunCreate {
     run_id?: string          // Optional custom ID
     name?: string            // Optional display name
     template: string         // Template filename
     config_content?: string  // Optional INI content override
   }

   // Paginated run list response
   export interface RunListResponse {
     runs: Run[]
     total: number
     limit: number
     offset: number
   }

Config Types
------------

.. code-block:: text

   export interface Template {
     name: string
     path: string
     description: string
   }

   export interface TemplateListResponse {
     templates: Template[]
   }

   export interface TemplateContent {
     name: string
     content: string
   }

Topology Types
--------------

.. code-block:: text

   export interface TopologyNode {
     id: number
     label: string
     x: number
     y: number
   }

   export interface TopologyLink {
     source: number
     target: number
     weight: number
   }

   export interface TopologyResponse {
     name: string
     nodes: TopologyNode[]
     links: TopologyLink[]
     metadata: {
       num_nodes: number
       num_links: number
     }
   }

   export interface TopologyListResponse {
     topologies: string[]
   }

Artifact Types
--------------

.. code-block:: text

   export interface Artifact {
     name: string
     path: string
     type: 'file' | 'directory'
     size: number
     modified: string
   }

   export interface ArtifactListResponse {
     items: Artifact[]
     path: string
   }

Codebase Types
--------------

.. code-block:: text

   export interface FileTreeNode {
     name: string
     type: 'file' | 'directory'
     path: string
     children?: FileTreeNode[]
   }

   export interface ModuleTreeResponse {
     tree: FileTreeNode
   }

   export interface FileContent {
     path: string
     content: string
     language: string
     size: number
   }

   export interface SearchResult {
     path: string
     type: 'file' | 'directory'
   }

System Types
------------

.. code-block:: text

   export interface HealthResponse {
     status: 'healthy' | 'unhealthy'
     version: string
     database: 'connected' | 'disconnected'
   }

----

Usage with React Query
======================

The API client functions are designed to work seamlessly with React Query.

Basic Query
-----------

.. code-block:: text

   import { useQuery } from '@tanstack/react-query'
   import { runsApi } from '@/api/client'

   function MyComponent() {
     const { data, isLoading, error } = useQuery({
       queryKey: ['runs'],
       queryFn: runsApi.list,
     })

     if (isLoading) return <Spinner />
     if (error) return <Error message={error.message} />

     return <RunList runs={data.runs} />
   }

Query with Parameters
---------------------

.. code-block:: text

   const { data } = useQuery({
     queryKey: ['runs', { status: 'RUNNING' }],
     queryFn: () => runsApi.list({ status: 'RUNNING' }),
   })

Dependent Query
---------------

.. code-block:: text

   const { data: run } = useQuery({
     queryKey: ['run', runId],
     queryFn: () => runsApi.get(runId),
     enabled: !!runId,  // Only run when runId is truthy
   })

Mutation
--------

.. code-block:: text

   import { useMutation, useQueryClient } from '@tanstack/react-query'

   function CreateRunButton() {
     const queryClient = useQueryClient()

     const mutation = useMutation({
       mutationFn: runsApi.create,
       onSuccess: () => {
         // Invalidate runs list to refetch
         queryClient.invalidateQueries({ queryKey: ['runs'] })
       },
     })

     return (
       <button
         onClick={() => mutation.mutate({ template: 'minimal.ini' })}
         disabled={mutation.isLoading}
       >
         {mutation.isLoading ? 'Creating...' : 'Create Run'}
       </button>
     )
   }

Polling
-------

.. code-block:: text

   const { data } = useQuery({
     queryKey: ['runs'],
     queryFn: runsApi.list,
     refetchInterval: 5000,  // Refetch every 5 seconds
   })
