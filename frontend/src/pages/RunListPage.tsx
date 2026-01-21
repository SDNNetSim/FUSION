import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Plus, RefreshCw } from 'lucide-react'
import { runsApi } from '@/api/client'
import { RunCard } from '@/components/runs/RunCard'

export function RunListPage() {
  const {
    data,
    isLoading,
    error,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ['runs'],
    queryFn: () => runsApi.list({ limit: 50 }),
    refetchInterval: 5000, // Poll every 5 seconds for status updates
  })

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Simulation Runs</h1>
          <p className="mt-1 text-sm text-gray-500">
            {data?.total ?? 0} total runs
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="btn-secondary flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </button>

          <Link to="/runs/new" className="btn-primary flex items-center gap-2">
            <Plus className="h-4 w-4" />
            New Run
          </Link>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
        </div>
      ) : error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
          Failed to load runs: {(error as Error).message}
        </div>
      ) : data?.runs.length === 0 ? (
        <div className="rounded-lg border-2 border-dashed border-gray-300 p-12 text-center">
          <h3 className="text-lg font-medium text-gray-900">No runs yet</h3>
          <p className="mt-1 text-sm text-gray-500">
            Get started by creating a new simulation run.
          </p>
          <Link to="/runs/new" className="btn-primary mt-4 inline-flex items-center gap-2">
            <Plus className="h-4 w-4" />
            Create Run
          </Link>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data?.runs.map((run) => (
            <RunCard key={run.id} run={run} />
          ))}
        </div>
      )}
    </div>
  )
}
