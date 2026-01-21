import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Download, Trash2, FolderOpen, FileText, Search, ChevronUp, ChevronDown, X } from 'lucide-react'
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
  const logsContainerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0)

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

  // Search matches
  const searchMatches = useMemo(() => {
    if (!searchTerm.trim()) return []
    const matches: number[] = []
    const term = searchTerm.toLowerCase()
    const lowerLogs = logs.toLowerCase()
    let idx = lowerLogs.indexOf(term)
    while (idx !== -1) {
      matches.push(idx)
      idx = lowerLogs.indexOf(term, idx + 1)
    }
    return matches
  }, [logs, searchTerm])

  // Reset match index when search term changes
  useEffect(() => {
    setCurrentMatchIndex(0)
  }, [searchTerm])

  // Scroll to current match
  const scrollToMatch = useCallback((index: number) => {
    if (searchMatches.length === 0 || !logsContainerRef.current) return
    const matchPosition = searchMatches[index]
    // Calculate approximate line number (assuming ~80 chars per line)
    const linesBefore = logs.slice(0, matchPosition).split('\n').length
    const lineHeight = 20 // approximate px per line
    const scrollTop = (linesBefore - 3) * lineHeight
    logsContainerRef.current.scrollTop = Math.max(0, scrollTop)
  }, [searchMatches, logs])

  const goToNextMatch = useCallback(() => {
    if (searchMatches.length === 0) return
    const nextIndex = (currentMatchIndex + 1) % searchMatches.length
    setCurrentMatchIndex(nextIndex)
    setAutoScroll(false)
    scrollToMatch(nextIndex)
  }, [currentMatchIndex, searchMatches.length, scrollToMatch])

  const goToPrevMatch = useCallback(() => {
    if (searchMatches.length === 0) return
    const prevIndex = (currentMatchIndex - 1 + searchMatches.length) % searchMatches.length
    setCurrentMatchIndex(prevIndex)
    setAutoScroll(false)
    scrollToMatch(prevIndex)
  }, [currentMatchIndex, searchMatches.length, scrollToMatch])

  // Highlight search matches in logs
  const highlightedLogs = useMemo(() => {
    if (!searchTerm.trim() || !logs) return logs
    const parts: (string | JSX.Element)[] = []
    let lastIndex = 0
    const term = searchTerm.toLowerCase()
    const lowerLogs = logs.toLowerCase()

    let matchNum = 0
    let idx = lowerLogs.indexOf(term)
    while (idx !== -1) {
      if (idx > lastIndex) {
        parts.push(logs.slice(lastIndex, idx))
      }
      const isCurrent = matchNum === currentMatchIndex
      parts.push(
        <mark
          key={idx}
          className={isCurrent
            ? 'bg-yellow-400 text-black rounded px-0.5'
            : 'bg-yellow-200 text-black rounded px-0.5'
          }
        >
          {logs.slice(idx, idx + searchTerm.length)}
        </mark>
      )
      lastIndex = idx + searchTerm.length
      matchNum++
      idx = lowerLogs.indexOf(term, lastIndex)
    }
    if (lastIndex < logs.length) {
      parts.push(logs.slice(lastIndex))
    }
    return parts
  }, [logs, searchTerm, currentMatchIndex])

  const clearSearch = useCallback(() => {
    setSearchTerm('')
    setCurrentMatchIndex(0)
  }, [])

  if (isLoading) {
    return <div className="flex items-center justify-center py-12 dark:text-gray-300">Loading...</div>
  }

  if (!run) {
    return <div className="text-red-600 dark:text-red-400">Run not found</div>
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
          className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to runs
        </Link>
      </div>

      {/* Header */}
      <div className="card mb-6 p-6 dark:bg-gray-800 dark:border-gray-700">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                {run.name || `Run ${run.id.slice(0, 6)}`}
              </h1>
              <RunStatusBadge status={run.status} />
            </div>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Template: {run.template} | ID: {run.id}
            </p>
            <div className="mt-2 text-xs text-gray-400 dark:text-gray-500">
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
            <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
              <span>Progress</span>
              <span>{run.progress.percent_complete?.toFixed(1) ?? 0}%</span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
              <div
                className="h-full bg-fusion-500 transition-all"
                style={{ width: `${run.progress.percent_complete ?? 0}%` }}
              />
            </div>
          </div>
        )}

        {run.error_message && (
          <div className="mt-4 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/20 dark:text-red-400">
            {run.error_message}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex gap-4">
          <button
            onClick={() => setActiveTab('logs')}
            className={`border-b-2 px-1 py-2 text-sm font-medium ${
              activeTab === 'logs'
                ? 'border-fusion-500 text-fusion-600 dark:text-fusion-400'
                : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            <FileText className="mr-2 inline h-4 w-4" />
            Logs
          </button>
          <button
            onClick={() => setActiveTab('artifacts')}
            className={`border-b-2 px-1 py-2 text-sm font-medium ${
              activeTab === 'artifacts'
                ? 'border-fusion-500 text-fusion-600 dark:text-fusion-400'
                : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
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
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              {/* Search Input */}
              <div className="relative flex-1 min-w-[200px] max-w-md">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search logs..."
                  className="w-full rounded-lg border border-gray-200 bg-white py-2 pl-10 pr-20 text-sm focus:border-fusion-500 focus:outline-none focus:ring-1 focus:ring-fusion-500 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100 dark:placeholder-gray-500"
                />
                {searchTerm && (
                  <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {searchMatches.length > 0
                        ? `${currentMatchIndex + 1}/${searchMatches.length}`
                        : '0/0'}
                    </span>
                    <button
                      onClick={goToPrevMatch}
                      disabled={searchMatches.length === 0}
                      className="p-0.5 text-gray-400 hover:text-gray-600 disabled:opacity-30 dark:hover:text-gray-300"
                      title="Previous match"
                    >
                      <ChevronUp className="h-4 w-4" />
                    </button>
                    <button
                      onClick={goToNextMatch}
                      disabled={searchMatches.length === 0}
                      className="p-0.5 text-gray-400 hover:text-gray-600 disabled:opacity-30 dark:hover:text-gray-300"
                      title="Next match"
                    >
                      <ChevronDown className="h-4 w-4" />
                    </button>
                    <button
                      onClick={clearSearch}
                      className="p-0.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                      title="Clear search"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>

              {/* Auto-scroll checkbox */}
              <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                  className="rounded border-gray-300 dark:border-gray-600"
                />
                Auto-scroll
              </label>
            </div>
            <div ref={logsContainerRef} className="log-viewer h-[500px]">
              <pre>{highlightedLogs || 'No logs yet...'}</pre>
              <div ref={logsEndRef} />
            </div>
          </div>
        )}

        {activeTab === 'artifacts' && (
          <div className="card dark:bg-gray-800 dark:border-gray-700">
            {artifactPath && (
              <div className="border-b border-gray-200 px-4 py-2 dark:border-gray-700">
                <button
                  onClick={handleNavigateUp}
                  className="text-sm text-fusion-600 hover:underline dark:text-fusion-400"
                >
                  .. (up)
                </button>
                <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">/{artifactPath}</span>
              </div>
            )}
            <div className="divide-y divide-gray-100 dark:divide-gray-700">
              {artifacts?.entries.length === 0 ? (
                <div className="px-4 py-8 text-center text-gray-500 dark:text-gray-400">
                  No files yet
                </div>
              ) : (
                artifacts?.entries.map((entry) => (
                  <button
                    key={entry.name}
                    onClick={() => handleArtifactClick(entry)}
                    className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    {entry.type === 'directory' ? (
                      <FolderOpen className="h-5 w-5 text-yellow-500" />
                    ) : (
                      <FileText className="h-5 w-5 text-gray-400 dark:text-gray-500" />
                    )}
                    <span className="flex-1 font-medium text-gray-900 dark:text-gray-100">{entry.name}</span>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      {formatBytes(entry.size_bytes)}
                    </span>
                    {entry.type === 'file' && (
                      <Download className="h-4 w-4 text-gray-400 dark:text-gray-500" />
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
