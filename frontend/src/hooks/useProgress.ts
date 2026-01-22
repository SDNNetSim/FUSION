import { useState, useEffect, useCallback } from 'react'

export interface ProgressEvent {
  type: string
  ts: string
  erlang?: number
  iteration?: number
  total_iterations?: number
  metrics?: {
    blocking_prob?: number
    [key: string]: unknown
  }
}

interface UseProgressOptions {
  enabled?: boolean
}

export function useProgress(runId: string | undefined, options: UseProgressOptions = {}) {
  const { enabled = true } = options
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [latestEvent, setLatestEvent] = useState<ProgressEvent | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const clearEvents = useCallback(() => {
    setEvents([])
    setLatestEvent(null)
  }, [])

  useEffect(() => {
    if (!runId || !enabled) return

    const eventSource = new EventSource(`/api/runs/${runId}/progress`)

    eventSource.onopen = () => {
      setIsConnected(true)
      setError(null)
    }

    eventSource.addEventListener('progress', (e) => {
      try {
        const event = JSON.parse(e.data) as ProgressEvent
        setEvents((prev) => [...prev, event])
        setLatestEvent(event)
      } catch {
        console.error('Failed to parse progress event:', e.data)
      }
    })

    eventSource.addEventListener('end', () => {
      setIsConnected(false)
      eventSource.close()
    })

    eventSource.addEventListener('error', (e) => {
      if (e instanceof MessageEvent && e.data) {
        setError(e.data)
      }
      setIsConnected(false)
    })

    eventSource.onerror = () => {
      setIsConnected(false)
      eventSource.close()
    }

    return () => {
      eventSource.close()
    }
  }, [runId, enabled])

  // Calculate derived values
  const percentComplete = latestEvent?.iteration && latestEvent?.total_iterations
    ? (latestEvent.iteration / latestEvent.total_iterations) * 100
    : null

  const blockingProbHistory = events
    .filter((e) => e.metrics?.blocking_prob !== undefined)
    .map((e) => ({
      iteration: e.iteration ?? 0,
      erlang: e.erlang ?? 0,
      blocking_prob: e.metrics?.blocking_prob ?? 0,
    }))

  return {
    events,
    latestEvent,
    isConnected,
    error,
    percentComplete,
    blockingProbHistory,
    clearEvents,
  }
}
