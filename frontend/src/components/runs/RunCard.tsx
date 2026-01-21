import { Link } from 'react-router-dom'
import { Clock, Trash2 } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { runsApi } from '@/api/client'
import { formatDate } from '@/lib/utils'
import { RunStatusBadge } from './RunStatusBadge'
import type { Run } from '@/api/types'

interface RunCardProps {
  run: Run
}

export function RunCard({ run }: RunCardProps) {
  const queryClient = useQueryClient()

  const cancelMutation = useMutation({
    mutationFn: () => runsApi.cancel(run.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })

  const handleCancel = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (confirm('Are you sure you want to cancel/delete this run?')) {
      cancelMutation.mutate()
    }
  }

  return (
    <Link
      to={`/runs/${run.id}`}
      className="card block p-4 transition-shadow hover:shadow-md dark:bg-gray-800 dark:border-gray-700"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-gray-900 dark:text-gray-100">
              {run.name || `Run ${run.id.slice(0, 6)}`}
            </h3>
            <RunStatusBadge status={run.status} />
          </div>

          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Template: {run.template}
          </p>

          <div className="mt-2 flex items-center gap-4 text-xs text-gray-400 dark:text-gray-500">
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {formatDate(run.created_at)}
            </span>
            <span className="font-mono">{run.id}</span>
          </div>

          {run.progress && run.status === 'RUNNING' && (
            <div className="mt-3">
              <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
                <span>Progress</span>
                <span>{run.progress.percent_complete?.toFixed(1) ?? 0}%</span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                <div
                  className="h-full bg-fusion-500 transition-all"
                  style={{ width: `${run.progress.percent_complete ?? 0}%` }}
                />
              </div>
            </div>
          )}

          {run.error_message && (
            <p className="mt-2 text-sm text-red-600 dark:text-red-400">{run.error_message}</p>
          )}
        </div>

        <button
          onClick={handleCancel}
          disabled={cancelMutation.isPending}
          className="ml-4 rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-red-600 dark:hover:bg-gray-700 dark:hover:text-red-400"
          title={run.status === 'RUNNING' ? 'Cancel run' : 'Delete run'}
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
    </Link>
  )
}
