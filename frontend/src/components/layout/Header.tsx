import { useQuery } from '@tanstack/react-query'
import { Activity } from 'lucide-react'
import { systemApi } from '@/api/client'

export function Header() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: systemApi.health,
    refetchInterval: 30000, // Check every 30 seconds
  })

  const isHealthy = health?.status === 'healthy'

  return (
    <header className="flex h-16 items-center justify-between border-b border-gray-200 bg-white px-6 dark:border-gray-700 dark:bg-gray-800">
      <div>
        {/* Breadcrumb or page title could go here */}
      </div>

      <div className="flex items-center gap-4">
        {/* API Status */}
        <div className="flex items-center gap-2 text-sm">
          <Activity
            className={cn(
              'h-4 w-4',
              isHealthy ? 'text-green-500' : 'text-red-500'
            )}
          />
          <span className={isHealthy ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
            {isHealthy ? 'API Connected' : 'API Disconnected'}
          </span>
        </div>
      </div>
    </header>
  )
}

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ')
}
