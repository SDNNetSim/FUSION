import { useState, useEffect, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Download, Trash2, FolderOpen, FileText } from 'lucide-react'
import { runsApi, artifactsApi } from '@/api/client'
import { RunStatusBadge } from '@/components/runs/RunStatusBadge'
import { formatDate, formatBytes } from '@/lib/utils'
import type { ArtifactEntry } from '@/api/types'

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'logs' | 'artifacts'>('logs')
  const [logs, setLogs] = useState('')
  const [artifactPath, setArtifactPath] = useState('')
  const logsEndRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  const { data: run, isLoading } = useQuery({
    queryKey: ['run', runId],
    queryFn: () => runsApi.get(runId!),
    enabled: !!runId,
    refetchInterval: 3000,
  })

  const { data: artifacts } = useQuery({
    queryKey: ['artifacts', runId, artifactPath],
    queryFn: () => artifactsApi.list(runId!, artifactPath),
    enabled: !!runId && activeTab === 'artifacts',
  })

  const cancelMutation = useMutation({
    mutationFn: () => runsApi.cancel(runId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['run', runId] })
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })

  // SSE for log streaming
  useEffect(() => {
    if (!runId || activeTab !== 'logs') return

    const eventSource = new EventSource(runsApi.getLogsUrl(runId, true))

    eventSource.addEventListener('log', (e) => {
      setLogs((prev) => prev + e.data + '\n')
    })

    eventSource.addEventListener('end', () => {
      eventSource.close()
    })

    eventSource.addEventListener('error', () => {
      eventSource.close()
    })

    return () => {
      eventSource.close()
    }
  }, [runId, activeTab])

  // Auto-scroll logs
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  if (isLoading) {
    return <div className="flex items-center justify-center py-12">Loading...</div>
  }

  if (!run) {
    return <div className="text-red-600">Run not found</div>
  }

  const handleArtifactClick = (entry: ArtifactEntry) => {
    if (entry.type === 'directory') {
      setArtifactPath(artifactPath ? `${artifactPath}/${entry.name}` : entry.name)
    } else {
      // Download file
      window.open(artifactsApi.getDownloadUrl(runId!, artifactPath ? `${artifactPath}/${entry.name}` : entry.name))
    }
  }

  const handleNavigateUp = () => {
    const parts = artifactPath.split('/')
    parts.pop()
    setArtifactPath(parts.join('/'))
  }

  return (
    <div>
      <div className="mb-6">
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to runs
        </Link>
      </div>

      {/* Header */}
      <div className="card mb-6 p-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-gray-900">
                {run.name || `Run ${run.id.slice(0, 6)}`}
              </h1>
              <RunStatusBadge status={run.status} />
            </div>
            <p className="mt-1 text-sm text-gray-500">
              Template: {run.template} | ID: {run.id}
            </p>
            <div className="mt-2 text-xs text-gray-400">
              Created: {formatDate(run.created_at)}
              {run.started_at && ` | Started: ${formatDate(run.started_at)}`}
              {run.completed_at && ` | Completed: ${formatDate(run.completed_at)}`}
            </div>
          </div>

          {(run.status === 'RUNNING' || run.status === 'COMPLETED' || run.status === 'FAILED') && (
            <button
              onClick={() => {
                if (confirm('Are you sure?')) cancelMutation.mutate()
              }}
              disabled={cancelMutation.isPending}
              className="btn-danger flex items-center gap-2"
            >
              <Trash2 className="h-4 w-4" />
              {run.status === 'RUNNING' ? 'Cancel' : 'Delete'}
            </button>
          )}
        </div>

        {run.progress && run.status === 'RUNNING' && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm text-gray-600">
              <span>Progress</span>
              <span>{run.progress.percent_complete?.toFixed(1) ?? 0}%</span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-gray-200">
              <div
                className="h-full bg-fusion-500 transition-all"
                style={{ width: `${run.progress.percent_complete ?? 0}%` }}
              />
            </div>
          </div>
        )}

        {run.error_message && (
          <div className="mt-4 rounded-md bg-red-50 p-3 text-sm text-red-700">
            {run.error_message}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex gap-4">
          <button
            onClick={() => setActiveTab('logs')}
            className={`border-b-2 px-1 py-2 text-sm font-medium ${
              activeTab === 'logs'
                ? 'border-fusion-500 text-fusion-600'
                : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
            }`}
          >
            <FileText className="mr-2 inline h-4 w-4" />
            Logs
          </button>
          <button
            onClick={() => setActiveTab('artifacts')}
            className={`border-b-2 px-1 py-2 text-sm font-medium ${
              activeTab === 'artifacts'
                ? 'border-fusion-500 text-fusion-600'
                : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
            }`}
          >
            <FolderOpen className="mr-2 inline h-4 w-4" />
            Artifacts
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-4">
        {activeTab === 'logs' && (
          <div>
            <div className="mb-2 flex items-center justify-between">
              <label className="flex items-center gap-2 text-sm text-gray-600">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                  className="rounded border-gray-300"
                />
                Auto-scroll
              </label>
            </div>
            <div className="log-viewer h-[500px]">
              <pre>{logs || 'No logs yet...'}</pre>
              <div ref={logsEndRef} />
            </div>
          </div>
        )}

        {activeTab === 'artifacts' && (
          <div className="card">
            {artifactPath && (
              <div className="border-b border-gray-200 px-4 py-2">
                <button
                  onClick={handleNavigateUp}
                  className="text-sm text-fusion-600 hover:underline"
                >
                  .. (up)
                </button>
                <span className="ml-2 text-sm text-gray-500">/{artifactPath}</span>
              </div>
            )}
            <div className="divide-y divide-gray-100">
              {artifacts?.entries.length === 0 ? (
                <div className="px-4 py-8 text-center text-gray-500">
                  No files yet
                </div>
              ) : (
                artifacts?.entries.map((entry) => (
                  <button
                    key={entry.name}
                    onClick={() => handleArtifactClick(entry)}
                    className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-gray-50"
                  >
                    {entry.type === 'directory' ? (
                      <FolderOpen className="h-5 w-5 text-yellow-500" />
                    ) : (
                      <FileText className="h-5 w-5 text-gray-400" />
                    )}
                    <span className="flex-1 font-medium text-gray-900">{entry.name}</span>
                    <span className="text-sm text-gray-500">
                      {formatBytes(entry.size_bytes)}
                    </span>
                    {entry.type === 'file' && (
                      <Download className="h-4 w-4 text-gray-400" />
                    )}
                  </button>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
