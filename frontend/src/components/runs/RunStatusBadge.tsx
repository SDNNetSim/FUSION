import { cn } from '@/lib/utils'
import type { Run } from '@/api/types'

interface RunStatusBadgeProps {
  status: Run['status']
}

const statusConfig = {
  PENDING: { label: 'Pending', className: 'bg-yellow-100 text-yellow-800' },
  RUNNING: { label: 'Running', className: 'bg-blue-100 text-blue-800' },
  COMPLETED: { label: 'Completed', className: 'bg-green-100 text-green-800' },
  FAILED: { label: 'Failed', className: 'bg-red-100 text-red-800' },
  CANCELLED: { label: 'Cancelled', className: 'bg-gray-100 text-gray-800' },
}

export function RunStatusBadge({ status }: RunStatusBadgeProps) {
  const config = statusConfig[status]

  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium',
        config.className
      )}
    >
      {status === 'RUNNING' && (
        <span className="mr-1.5 h-2 w-2 animate-pulse rounded-full bg-blue-500" />
      )}
      {config.label}
    </span>
  )
}
