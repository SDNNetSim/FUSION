import { Sun, Moon, Monitor } from 'lucide-react'
import { useUI } from '@/stores/ui'
import { cn } from '@/lib/utils'

export function SettingsPage() {
  const { theme, setTheme } = useUI()

  const themeOptions = [
    { value: 'light' as const, label: 'Light', icon: Sun },
    { value: 'dark' as const, label: 'Dark', icon: Moon },
    { value: 'system' as const, label: 'System', icon: Monitor },
  ]

  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Settings</h1>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Configure your FUSION GUI preferences.
      </p>

      <div className="mt-8 space-y-8">
        {/* Theme Section */}
        <div className="card p-6 dark:bg-gray-800 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100">Appearance</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Choose how FUSION GUI looks to you.
          </p>

          <div className="mt-4 grid grid-cols-3 gap-3">
            {themeOptions.map((option) => (
              <button
                key={option.value}
                onClick={() => setTheme(option.value)}
                className={cn(
                  'flex flex-col items-center gap-2 rounded-lg border-2 p-4 transition-colors',
                  theme === option.value
                    ? 'border-fusion-500 bg-fusion-50 dark:bg-fusion-900/20'
                    : 'border-gray-200 hover:border-gray-300 dark:border-gray-600 dark:hover:border-gray-500'
                )}
              >
                <option.icon
                  className={cn(
                    'h-6 w-6',
                    theme === option.value
                      ? 'text-fusion-600'
                      : 'text-gray-400 dark:text-gray-500'
                  )}
                />
                <span
                  className={cn(
                    'text-sm font-medium',
                    theme === option.value
                      ? 'text-fusion-700 dark:text-fusion-400'
                      : 'text-gray-700 dark:text-gray-300'
                  )}
                >
                  {option.label}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* About Section */}
        <div className="card p-6 dark:bg-gray-800 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100">About</h2>
          <dl className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-gray-500 dark:text-gray-400">Version</dt>
              <dd className="text-gray-900 dark:text-gray-100">1.0.0</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500 dark:text-gray-400">API</dt>
              <dd className="text-gray-900 dark:text-gray-100">http://127.0.0.1:8765</dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  )
}
