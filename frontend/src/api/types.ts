/**
 * API type definitions matching the backend Pydantic schemas.
 */

export interface RunProgress {
  current_erlang: number | null
  total_erlangs: number | null
  current_iteration: number | null
  total_iterations: number | null
  percent_complete: number | null
}

export interface Run {
  id: string
  name: string | null
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'
  template: string
  created_at: string
  started_at: string | null
  completed_at: string | null
  error_message: string | null
  progress: RunProgress | null
}

export interface RunCreate {
  name?: string
  template?: string
  config?: Record<string, unknown>
}

export interface RunListResponse {
  runs: Run[]
  total: number
  limit: number
  offset: number
}

export interface TemplateInfo {
  name: string
  description: string | null
  path: string
}

export interface TemplateListResponse {
  templates: TemplateInfo[]
}

export interface TemplateContent {
  name: string
  content: string
}

export interface ArtifactEntry {
  name: string
  type: 'file' | 'directory'
  size_bytes: number | null
  modified_at: string
}

export interface ArtifactListResponse {
  path: string
  entries: ArtifactEntry[]
}

export interface HealthResponse {
  status: string
}

export interface VersionResponse {
  api_version: string
  fusion_version: string
}

// Topology types
export interface TopologyNode {
  id: string
  label: string
  x: number
  y: number
  type: string
}

export interface TopologyLink {
  id: string
  source: string
  target: string
  length_km: number
  utilization: number
}

export interface TopologyResponse {
  name: string
  nodes: TopologyNode[]
  links: TopologyLink[]
}

export interface TopologyListItem {
  name: string
  node_count: number
  link_count: number
}

export interface TopologyListResponse {
  topologies: TopologyListItem[]
}
