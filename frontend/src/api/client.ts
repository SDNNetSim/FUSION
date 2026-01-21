import axios from 'axios'
import type {
  Run,
  RunCreate,
  RunListResponse,
  TemplateListResponse,
  TemplateContent,
  ArtifactListResponse,
  HealthResponse,
  VersionResponse,
} from './types'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Runs API
export const runsApi = {
  list: async (params?: { status?: string; limit?: number; offset?: number }) => {
    const { data } = await api.get<RunListResponse>('/runs', { params })
    return data
  },

  get: async (runId: string) => {
    const { data } = await api.get<Run>(`/runs/${runId}`)
    return data
  },

  create: async (run: RunCreate) => {
    const { data } = await api.post<Run>('/runs', run)
    return data
  },

  cancel: async (runId: string) => {
    const { data } = await api.delete<Run>(`/runs/${runId}`)
    return data
  },

  getLogsUrl: (runId: string, fromStart = true) => {
    return `/api/runs/${runId}/logs?from_start=${fromStart}`
  },
}

// Configs API
export const configsApi = {
  listTemplates: async () => {
    const { data } = await api.get<TemplateListResponse>('/configs/templates')
    return data
  },

  getTemplate: async (name: string) => {
    const { data } = await api.get<TemplateContent>(`/configs/templates/${name}`)
    return data
  },
}

// Artifacts API
export const artifactsApi = {
  list: async (runId: string, path = '') => {
    const { data } = await api.get<ArtifactListResponse>(`/runs/${runId}/artifacts`, {
      params: path ? { path } : undefined,
    })
    return data
  },

  getDownloadUrl: (runId: string, filePath: string) => {
    return `/api/runs/${runId}/artifacts/${filePath}`
  },
}

// System API
export const systemApi = {
  health: async () => {
    const { data } = await api.get<HealthResponse>('/health')
    return data
  },

  version: async () => {
    const { data } = await api.get<VersionResponse>('/version')
    return data
  },
}

export default api
