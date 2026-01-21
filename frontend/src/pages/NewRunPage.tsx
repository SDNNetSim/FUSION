import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Play, ArrowLeft } from 'lucide-react'
import { runsApi, configsApi } from '@/api/client'
import { Link } from 'react-router-dom'

export function NewRunPage() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [template, setTemplate] = useState('default')

  const { data: templatesData, isLoading: templatesLoading } = useQuery({
    queryKey: ['templates'],
    queryFn: configsApi.listTemplates,
  })

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
          className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to runs
        </Link>
      </div>

      <div className="card p-6">
        <h1 className="text-xl font-bold text-gray-900">Create New Run</h1>
        <p className="mt-1 text-sm text-gray-500">
          Start a new simulation with a configuration template.
        </p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700">
              Run Name (optional)
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Simulation"
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-fusion-500 focus:outline-none focus:ring-1 focus:ring-fusion-500"
            />
          </div>

          <div>
            <label htmlFor="template" className="block text-sm font-medium text-gray-700">
              Configuration Template
            </label>
            <select
              id="template"
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
              disabled={templatesLoading}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-fusion-500 focus:outline-none focus:ring-1 focus:ring-fusion-500"
            >
              {templatesLoading ? (
                <option>Loading templates...</option>
              ) : (
                templatesData?.templates.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name}
                    {t.description && ` - ${t.description}`}
                  </option>
                ))
              )}
            </select>
          </div>

          {createMutation.error && (
            <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">
              {(createMutation.error as Error).message}
            </div>
          )}

          <div className="flex justify-end gap-3 pt-4">
            <Link to="/" className="btn-secondary">
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
