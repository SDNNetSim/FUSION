import { useState, useEffect, useRef } from 'react'
import { useSearchParams, Link, useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { ArrowLeft, Check, AlertTriangle, Save, FileCode, ChevronDown, FileText } from 'lucide-react'
import Editor from '@monaco-editor/react'
import { configsApi } from '@/api/client'
import { useUI } from '@/stores/ui'
import axios from 'axios'

interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

export function ConfigEditorPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()
  const templateName = searchParams.get('template') || 'default'
  const { isDark } = useUI()

  const [content, setContent] = useState('')
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const { data: templateList } = useQuery({
    queryKey: ['templates'],
    queryFn: configsApi.listTemplates,
  })

  const { data: templateData, isLoading } = useQuery({
    queryKey: ['template', templateName],
    queryFn: () => configsApi.getTemplate(templateName),
  })

  useEffect(() => {
    if (templateData?.content) {
      setContent(templateData.content)
      setValidation(null)
    }
  }, [templateData])

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

  const handleTemplateChange = (name: string) => {
    setSearchParams({ template: name })
    setDropdownOpen(false)
  }

  if (isLoading) {
    return <div className="flex items-center justify-center py-12 dark:text-gray-300">Loading...</div>
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            to="/"
            className="flex items-center justify-center rounded-lg p-2 text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-700"
          >
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Config Editor
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Edit simulation configuration templates
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Template Selector */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-4 py-2.5 text-sm font-medium text-gray-700 shadow-sm transition-all hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
            >
              <FileCode className="h-4 w-4 text-fusion-500" />
              {templateName}
              <ChevronDown className={`h-4 w-4 text-gray-400 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
            </button>

            {/* Dropdown Menu */}
            {dropdownOpen && templateList && (
              <div className="absolute right-0 top-full z-50 mt-2 w-72 rounded-lg border border-gray-200 bg-white py-1 shadow-lg dark:border-gray-700 dark:bg-gray-800">
                <div className="px-3 py-2 text-xs font-medium uppercase text-gray-500 dark:text-gray-400">
                  Available Templates
                </div>
                {templateList.templates.map((t) => (
                  <button
                    key={t.name}
                    onClick={() => handleTemplateChange(t.name)}
                    className={`flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                      templateName === t.name
                        ? 'bg-fusion-50 dark:bg-fusion-900/20'
                        : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                  >
                    <FileText className={`h-4 w-4 ${templateName === t.name ? 'text-fusion-500' : 'text-gray-400'}`} />
                    <div className="flex-1 min-w-0">
                      <div className={`text-sm font-medium truncate ${
                        templateName === t.name
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
                    {templateName === t.name && (
                      <Check className="h-4 w-4 text-fusion-600 flex-shrink-0" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="h-8 w-px bg-gray-200 dark:bg-gray-700" />

          <button
            onClick={handleValidate}
            disabled={validateMutation.isPending}
            className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-4 py-2.5 text-sm font-medium text-gray-700 shadow-sm transition-all hover:bg-gray-50 disabled:opacity-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
          >
            <Check className="h-4 w-4" />
            {validateMutation.isPending ? 'Validating...' : 'Validate'}
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 rounded-lg bg-fusion-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition-all hover:bg-fusion-700"
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
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-100 dark:bg-green-800">
                <Check className="h-4 w-4 text-green-600 dark:text-green-400" />
              </div>
            ) : (
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-red-100 dark:bg-red-800">
                <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
              </div>
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
            <ul className="mt-3 space-y-1 text-sm text-red-600 dark:text-red-400">
              {validation.errors.map((error, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-red-400">-</span>
                  {error}
                </li>
              ))}
            </ul>
          )}

          {validation.warnings.length > 0 && (
            <ul className="mt-3 space-y-1 text-sm text-yellow-600 dark:text-yellow-400">
              {validation.warnings.map((warning, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-yellow-400">-</span>
                  {warning}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Editor */}
      <div className="flex-1 overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
        <Editor
          height="100%"
          defaultLanguage="ini"
          value={content}
          onChange={(value) => setContent(value || '')}
          theme={isDark ? 'vs-dark' : 'light'}
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            lineNumbers: 'on',
            scrollBeyondLastLine: false,
            wordWrap: 'on',
            automaticLayout: true,
            tabSize: 2,
            padding: { top: 16, bottom: 16 },
          }}
          loading={
            <div className="flex h-full items-center justify-center bg-white dark:bg-gray-800">
              <span className="text-gray-500 dark:text-gray-400">Loading editor...</span>
            </div>
          }
        />
      </div>
    </div>
  )
}
