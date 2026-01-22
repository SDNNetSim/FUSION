import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Play, ArrowLeft, FileCode, ChevronDown, Check } from 'lucide-react'
import { runsApi, configsApi } from '@/api/client'
import { Link } from 'react-router-dom'

export function NewRunPage() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [template, setTemplate] = useState('default')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const { data: templatesData, isLoading: templatesLoading } = useQuery({
    queryKey: ['templates'],
    queryFn: configsApi.listTemplates,
  })

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selectedTemplate = templatesData?.templates.find(t => t.name === template)

  const createMutation = useMutation({
    mutationFn: runsApi.create,
    onSuccess: (run) => {
      navigate(`/runs/${run.id}`)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createMutation.mutate({
      name: name || undefined,
      template,
    })
  }

  return (
    <div className="mx-auto max-w-2xl">
      <div className="mb-6">
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to runs
        </Link>
      </div>

      <div className="card p-6 dark:bg-gray-800 dark:border-gray-700">
        <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">Create New Run</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Start a new simulation with a configuration template.
        </p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Run Name (optional)
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Simulation"
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-fusion-500 focus:outline-none focus:ring-1 focus:ring-fusion-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100 dark:placeholder-gray-400"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Configuration Template
            </label>
            <div className="relative" ref={dropdownRef}>
              <button
                type="button"
                onClick={() => setDropdownOpen(!dropdownOpen)}
                disabled={templatesLoading}
                className="flex w-full items-center gap-3 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-left shadow-sm transition-all hover:border-gray-400 focus:border-fusion-500 focus:outline-none focus:ring-2 focus:ring-fusion-500/20 dark:border-gray-600 dark:bg-gray-700 dark:hover:border-gray-500"
              >
                <FileCode className="h-5 w-5 text-fusion-500 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {templatesLoading ? 'Loading...' : template}
                  </div>
                  {selectedTemplate?.description && (
                    <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                      {selectedTemplate.description}
                    </div>
                  )}
                </div>
                <ChevronDown className={`h-4 w-4 text-gray-400 flex-shrink-0 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu */}
              {dropdownOpen && templatesData && (
                <div className="absolute left-0 right-0 top-full z-50 mt-2 max-h-60 overflow-y-auto rounded-lg border border-gray-200 bg-white py-1 shadow-lg dark:border-gray-700 dark:bg-gray-800">
                  <div className="px-3 py-2 text-xs font-medium uppercase text-gray-500 dark:text-gray-400">
                    Available Templates
                  </div>
                  {templatesData.templates.map((t) => (
                    <button
                      type="button"
                      key={t.name}
                      onClick={() => {
                        setTemplate(t.name)
                        setDropdownOpen(false)
                      }}
                      className={`flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                        template === t.name
                          ? 'bg-fusion-50 dark:bg-fusion-900/20'
                          : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                    >
                      <FileCode className={`h-4 w-4 flex-shrink-0 ${template === t.name ? 'text-fusion-500' : 'text-gray-400'}`} />
                      <div className="flex-1 min-w-0">
                        <div className={`text-sm font-medium truncate ${
                          template === t.name
                            ? 'text-fusion-700 dark:text-fusion-400'
                            : 'text-gray-900 dark:text-gray-100'
                        }`}>
                          {t.name}
                        </div>
                        {t.description && (
                          <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                            {t.description}
                          </div>
                        )}
                      </div>
                      {template === t.name && (
                        <Check className="h-4 w-4 text-fusion-600 flex-shrink-0" />
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {createMutation.error && (
            <div className="rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/20 dark:text-red-400">
              {(createMutation.error as Error).message}
            </div>
          )}

          <div className="flex justify-end gap-3 pt-4">
            <Link to="/" className="btn-secondary dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600">
              Cancel
            </Link>
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="btn-primary flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              {createMutation.isPending ? 'Starting...' : 'Start Run'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
