import { useState, useEffect } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { ArrowLeft, Check, AlertTriangle, Save } from 'lucide-react'
import { configsApi } from '@/api/client'
import axios from 'axios'

interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

export function ConfigEditorPage() {
  const [searchParams] = useSearchParams()
  const templateName = searchParams.get('template') || 'default'

  const [content, setContent] = useState('')
  const [validation, setValidation] = useState<ValidationResult | null>(null)

  const { data: templateData, isLoading } = useQuery({
    queryKey: ['template', templateName],
    queryFn: () => configsApi.getTemplate(templateName),
  })

  useEffect(() => {
    if (templateData?.content) {
      setContent(templateData.content)
    }
  }, [templateData])

  const validateMutation = useMutation({
    mutationFn: async (configContent: string) => {
      const { data } = await axios.post<ValidationResult>('/api/configs/validate', {
        content: configContent,
      })
      return data
    },
    onSuccess: (result) => {
      setValidation(result)
    },
  })

  const handleValidate = () => {
    validateMutation.mutate(content)
  }

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${templateName}_modified.ini`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (isLoading) {
    return <div className="flex items-center justify-center py-12">Loading...</div>
  }

  return (
    <div className="flex h-full flex-col">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <Link
            to="/"
            className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to runs
          </Link>
          <h1 className="mt-2 text-xl font-bold text-gray-900 dark:text-gray-100">
            Config Editor: {templateName}
          </h1>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleValidate}
            disabled={validateMutation.isPending}
            className="btn-secondary flex items-center gap-2"
          >
            <Check className="h-4 w-4" />
            {validateMutation.isPending ? 'Validating...' : 'Validate'}
          </button>
          <button
            onClick={handleDownload}
            className="btn-primary flex items-center gap-2"
          >
            <Save className="h-4 w-4" />
            Download
          </button>
        </div>
      </div>

      {/* Validation Results */}
      {validation && (
        <div
          className={`mb-4 rounded-lg border p-4 ${
            validation.valid
              ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
              : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
          }`}
        >
          <div className="flex items-center gap-2">
            {validation.valid ? (
              <Check className="h-5 w-5 text-green-600" />
            ) : (
              <AlertTriangle className="h-5 w-5 text-red-600" />
            )}
            <span
              className={`font-medium ${
                validation.valid ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'
              }`}
            >
              {validation.valid ? 'Configuration is valid' : 'Configuration has errors'}
            </span>
          </div>

          {validation.errors.length > 0 && (
            <ul className="mt-2 space-y-1 text-sm text-red-600 dark:text-red-400">
              {validation.errors.map((error, i) => (
                <li key={i}>- {error}</li>
              ))}
            </ul>
          )}

          {validation.warnings.length > 0 && (
            <ul className="mt-2 space-y-1 text-sm text-yellow-600 dark:text-yellow-400">
              {validation.warnings.map((warning, i) => (
                <li key={i}>- {warning}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Editor */}
      <div className="flex-1 rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="h-full w-full resize-none rounded-lg bg-transparent p-4 font-mono text-sm text-gray-900 focus:outline-none dark:text-gray-100"
          spellCheck={false}
          placeholder="# Configuration content..."
        />
      </div>
    </div>
  )
}
