import { useState, useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ChevronRight,
  Search,
  Code2,
  Box,
  X,
  FileCode,
  Cpu,
  Network,
  Database,
  Layers,
  Terminal,
  Brain,
  Shield,
  Cog,
  BookOpen,
  Play,
  ArrowLeft,
  ExternalLink,
  ChevronDown,
  Sparkles,
} from 'lucide-react'
import Editor from '@monaco-editor/react'
import { codebaseApi } from '@/api/client'
import { useUI } from '@/stores/ui'
import type { ModuleNode, SearchResult } from '@/api/types'

// Module metadata for the visual architecture
const MODULE_INFO: Record<string, {
  icon: typeof Cpu
  color: string
  bgColor: string
  description: string
  highlights: string[]
}> = {
  core: {
    icon: Cpu,
    color: 'text-blue-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    description: 'The heart of FUSION - simulation engine, event processing, and request handling.',
    highlights: ['SimulationEngine', 'Event Processing', 'Request Lifecycle'],
  },
  modules: {
    icon: Layers,
    color: 'text-purple-600',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
    description: 'Pluggable algorithm modules for routing, spectrum assignment, and AI/ML integration.',
    highlights: ['Routing Algorithms', 'Spectrum Assignment', 'Reinforcement Learning'],
  },
  io: {
    icon: Database,
    color: 'text-green-600',
    bgColor: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
    description: 'Input/output handling - topology loading, results generation, and data persistence.',
    highlights: ['Topology Parser', 'Results Export', 'Configuration'],
  },
  cli: {
    icon: Terminal,
    color: 'text-orange-600',
    bgColor: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800',
    description: 'Command-line interface for running simulations and managing experiments.',
    highlights: ['Run Commands', 'Parameter Parsing', 'Batch Processing'],
  },
  domain: {
    icon: Box,
    color: 'text-cyan-600',
    bgColor: 'bg-cyan-50 dark:bg-cyan-900/20 border-cyan-200 dark:border-cyan-800',
    description: 'Domain models and data structures representing network state and requests.',
    highlights: ['Network State', 'Lightpath Model', 'Spectrum Management'],
  },
  interfaces: {
    icon: Cog,
    color: 'text-gray-600',
    bgColor: 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700',
    description: 'Abstract interfaces and factories for pluggable algorithm architecture.',
    highlights: ['Algorithm Interfaces', 'Factory Pattern', 'Plugin System'],
  },
  api: {
    icon: Network,
    color: 'text-pink-600',
    bgColor: 'bg-pink-50 dark:bg-pink-900/20 border-pink-200 dark:border-pink-800',
    description: 'REST API for the web GUI - run management, streaming, and visualization.',
    highlights: ['FastAPI Routes', 'SSE Streaming', 'WebSocket Support'],
  },
  configs: {
    icon: FileCode,
    color: 'text-amber-600',
    bgColor: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800',
    description: 'Configuration templates and INI file management for simulations.',
    highlights: ['Templates', 'Validation', 'Defaults'],
  },
}

// Guided tour steps
const TOUR_STEPS = [
  {
    module: null,
    title: 'Welcome to FUSION',
    description: 'FUSION is an optical network simulator for Software Defined Elastic Optical Networks (SD-EONs). Let\'s explore how it\'s organized.',
  },
  {
    module: 'core',
    title: 'The Simulation Engine',
    description: 'The core module contains the main simulation engine. It processes network requests using discrete event simulation, managing the lifecycle of each connection request.',
  },
  {
    module: 'modules',
    title: 'Algorithm Modules',
    description: 'This is where the magic happens! Routing algorithms, spectrum assignment strategies, and reinforcement learning agents live here. Each algorithm is pluggable and swappable.',
  },
  {
    module: 'io',
    title: 'Input/Output',
    description: 'The io module handles loading network topologies, reading configurations, and exporting simulation results. It supports multiple file formats.',
  },
  {
    module: 'domain',
    title: 'Domain Models',
    description: 'Data structures that represent the network state, lightpaths, spectrum slots, and other domain concepts. The building blocks of the simulation.',
  },
  {
    module: 'cli',
    title: 'Command Line Interface',
    description: 'Run simulations from the terminal with flexible parameters. Supports batch processing and experiment automation.',
  },
]

type ViewMode = 'architecture' | 'module' | 'code'

export function CodebaseExplorerPage() {
  const { isDark } = useUI()
  const [viewMode, setViewMode] = useState<ViewMode>('architecture')
  const [selectedModule, setSelectedModule] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [showSearch, setShowSearch] = useState(false)
  const [tourActive, setTourActive] = useState(false)
  const [tourStep, setTourStep] = useState(0)

  const { data: tree } = useQuery({
    queryKey: ['codebase-tree'],
    queryFn: codebaseApi.getTree,
  })

  const { data: fileContent, isLoading: fileLoading } = useQuery({
    queryKey: ['codebase-file', selectedFile],
    queryFn: () => codebaseApi.getFile(selectedFile!),
    enabled: !!selectedFile,
  })

  // Get module node from tree
  const getModuleNode = useCallback((moduleName: string): ModuleNode | null => {
    if (!tree) return null
    return tree.root.children.find(c => c.name === moduleName) || null
  }, [tree])

  const selectedModuleNode = selectedModule ? getModuleNode(selectedModule) : null
  const moduleInfo = selectedModule ? MODULE_INFO[selectedModule] : null

  // Count files in a module
  const countFiles = useCallback((node: ModuleNode): number => {
    if (node.type === 'module') return 1
    return node.children.reduce((sum, child) => sum + countFiles(child), 0)
  }, [])

  // Search handler
  const handleSearch = useCallback(async (query: string) => {
    setSearchQuery(query)
    if (query.length < 2) {
      setSearchResults([])
      return
    }
    try {
      const results = await codebaseApi.search(query)
      setSearchResults(results)
    } catch {
      setSearchResults([])
    }
  }, [])

  // Navigation handlers
  const goToModule = useCallback((moduleName: string) => {
    setSelectedModule(moduleName)
    setViewMode('module')
    setSelectedFile(null)
  }, [])

  const goToFile = useCallback((path: string) => {
    setSelectedFile(path)
    setViewMode('code')
    // Extract module from path
    const parts = path.split('/')
    if (parts.length > 1) {
      setSelectedModule(parts[1])
    }
  }, [])

  const goBack = useCallback(() => {
    if (viewMode === 'code') {
      setViewMode('module')
      setSelectedFile(null)
    } else if (viewMode === 'module') {
      setViewMode('architecture')
      setSelectedModule(null)
    }
  }, [viewMode])

  // Tour handlers
  const startTour = useCallback(() => {
    setTourActive(true)
    setTourStep(0)
    setViewMode('architecture')
    setSelectedModule(null)
  }, [])

  const nextTourStep = useCallback(() => {
    const nextStep = tourStep + 1
    if (nextStep >= TOUR_STEPS.length) {
      setTourActive(false)
      return
    }
    setTourStep(nextStep)
    const step = TOUR_STEPS[nextStep]
    if (step.module) {
      goToModule(step.module)
    } else {
      setViewMode('architecture')
      setSelectedModule(null)
    }
  }, [tourStep, goToModule])

  const endTour = useCallback(() => {
    setTourActive(false)
  }, [])

  // Modules to display in architecture view
  const mainModules = useMemo(() => {
    if (!tree) return []
    return tree.root.children
      .filter(c => c.type === 'package' && MODULE_INFO[c.name])
      .map(c => ({
        ...c,
        info: MODULE_INFO[c.name],
        fileCount: countFiles(c),
      }))
  }, [tree, countFiles])

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {viewMode !== 'architecture' && (
            <button
              onClick={goBack}
              className="flex items-center gap-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
          )}
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {viewMode === 'architecture' && 'Explore FUSION'}
              {viewMode === 'module' && selectedModule && (
                <span className="flex items-center gap-2">
                  {moduleInfo && <moduleInfo.icon className={`h-6 w-6 ${moduleInfo.color}`} />}
                  {selectedModule}
                </span>
              )}
              {viewMode === 'code' && selectedFile && (
                <span className="text-lg">{selectedFile.split('/').pop()}</span>
              )}
            </h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {viewMode === 'architecture' && 'Click on a module to explore its contents'}
              {viewMode === 'module' && moduleInfo?.description}
              {viewMode === 'code' && selectedFile}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Tour Button */}
          {viewMode === 'architecture' && !tourActive && (
            <button
              onClick={startTour}
              className="flex items-center gap-2 rounded-lg bg-fusion-600 px-4 py-2 text-sm font-medium text-white hover:bg-fusion-700"
            >
              <Sparkles className="h-4 w-4" />
              Take a Tour
            </button>
          )}

          {/* Search Toggle */}
          <button
            onClick={() => setShowSearch(!showSearch)}
            className={`rounded-lg p-2 ${showSearch ? 'bg-fusion-100 text-fusion-600 dark:bg-fusion-900/30' : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
          >
            <Search className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Search Bar */}
      {showSearch && (
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search files, classes, functions..."
            className="w-full rounded-lg border border-gray-200 bg-white py-3 pl-10 pr-4 text-sm focus:border-fusion-500 focus:outline-none focus:ring-1 focus:ring-fusion-500 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
            autoFocus
          />
          {searchQuery && (
            <button
              onClick={() => { setSearchQuery(''); setSearchResults([]) }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              <X className="h-4 w-4" />
            </button>
          )}

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="absolute left-0 right-0 top-full z-50 mt-1 max-h-80 overflow-auto rounded-lg border border-gray-200 bg-white shadow-lg dark:border-gray-700 dark:bg-gray-800">
              {searchResults.map((result, i) => (
                <button
                  key={`${result.path}-${i}`}
                  onClick={() => {
                    goToFile(result.path)
                    setShowSearch(false)
                    setSearchQuery('')
                    setSearchResults([])
                  }}
                  className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700"
                >
                  {result.type === 'file' ? (
                    <FileCode className="h-5 w-5 text-blue-500" />
                  ) : result.type === 'class' ? (
                    <Box className="h-5 w-5 text-purple-500" />
                  ) : (
                    <Code2 className="h-5 w-5 text-green-500" />
                  )}
                  <div>
                    <div className="font-medium text-gray-900 dark:text-gray-100">{result.name}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">{result.path}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tour Overlay */}
      {tourActive && (
        <div className="mb-4 rounded-lg border-2 border-fusion-500 bg-fusion-50 p-4 dark:bg-fusion-900/20">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-fusion-600" />
                <span className="text-sm font-medium text-fusion-600">
                  Step {tourStep + 1} of {TOUR_STEPS.length}
                </span>
              </div>
              <h3 className="mt-2 text-lg font-bold text-gray-900 dark:text-gray-100">
                {TOUR_STEPS[tourStep].title}
              </h3>
              <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                {TOUR_STEPS[tourStep].description}
              </p>
            </div>
            <button onClick={endTour} className="text-gray-400 hover:text-gray-600">
              <X className="h-5 w-5" />
            </button>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <button
              onClick={endTour}
              className="rounded px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
            >
              End Tour
            </button>
            <button
              onClick={nextTourStep}
              className="flex items-center gap-1 rounded bg-fusion-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-fusion-700"
            >
              {tourStep < TOUR_STEPS.length - 1 ? 'Next' : 'Finish'}
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        {/* Architecture View */}
        {viewMode === 'architecture' && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {mainModules.map((module) => (
              <button
                key={module.name}
                onClick={() => goToModule(module.name)}
                className={`group relative rounded-xl border-2 p-5 text-left transition-all hover:shadow-lg hover:-translate-y-1 ${module.info.bgColor} ${tourActive && TOUR_STEPS[tourStep].module === module.name ? 'ring-2 ring-fusion-500 ring-offset-2' : ''}`}
              >
                <div className="flex items-start justify-between">
                  <module.info.icon className={`h-8 w-8 ${module.info.color}`} />
                  <ChevronRight className="h-5 w-5 text-gray-400 transition-transform group-hover:translate-x-1" />
                </div>
                <h3 className="mt-3 text-lg font-bold text-gray-900 dark:text-gray-100">
                  {module.name}
                </h3>
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-300 line-clamp-2">
                  {module.info.description}
                </p>
                <div className="mt-3 flex flex-wrap gap-1">
                  {module.info.highlights.slice(0, 2).map((h) => (
                    <span
                      key={h}
                      className="rounded-full bg-white/60 px-2 py-0.5 text-xs text-gray-700 dark:bg-gray-800/60 dark:text-gray-300"
                    >
                      {h}
                    </span>
                  ))}
                </div>
                <div className="mt-3 text-xs text-gray-500 dark:text-gray-400">
                  {module.fileCount} files
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Module View */}
        {viewMode === 'module' && selectedModuleNode && moduleInfo && (
          <div>
            {/* Module Header Card */}
            <div className={`mb-6 rounded-xl border-2 p-6 ${moduleInfo.bgColor}`}>
              <div className="flex items-start gap-4">
                <moduleInfo.icon className={`h-12 w-12 ${moduleInfo.color}`} />
                <div className="flex-1">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                    {selectedModule}
                  </h2>
                  <p className="mt-1 text-gray-600 dark:text-gray-300">
                    {moduleInfo.description}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {moduleInfo.highlights.map((h) => (
                      <span
                        key={h}
                        className="rounded-full bg-white/80 px-3 py-1 text-sm font-medium text-gray-700 dark:bg-gray-800/80 dark:text-gray-300"
                      >
                        {h}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Submodules and Files */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {selectedModuleNode.children.map((child) => (
                <button
                  key={child.path}
                  onClick={() => {
                    if (child.type === 'module') {
                      goToFile(child.path)
                    } else {
                      // For packages, show their contents
                      setSelectedModule(child.path.replace('fusion/', ''))
                    }
                  }}
                  className="group flex items-start gap-3 rounded-lg border border-gray-200 bg-white p-4 text-left transition-all hover:border-gray-300 hover:shadow dark:border-gray-700 dark:bg-gray-800 dark:hover:border-gray-600"
                >
                  {child.type === 'package' ? (
                    <Layers className="h-6 w-6 text-purple-500" />
                  ) : (
                    <FileCode className="h-6 w-6 text-blue-500" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-900 dark:text-gray-100 truncate">
                        {child.name}
                      </span>
                      {child.type === 'package' && (
                        <ChevronRight className="h-4 w-4 text-gray-400 group-hover:translate-x-0.5 transition-transform" />
                      )}
                    </div>
                    {child.description && (
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400 line-clamp-2">
                        {child.description}
                      </p>
                    )}
                    <div className="mt-2 text-xs text-gray-400">
                      {child.type === 'package' ? `${countFiles(child)} files` : 'Python module'}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Code View */}
        {viewMode === 'code' && (
          <div className="flex h-full gap-4">
            {/* Code Editor */}
            <div className="flex-1 overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
              {fileLoading ? (
                <div className="flex h-full items-center justify-center bg-white dark:bg-gray-800">
                  <span className="text-gray-500">Loading...</span>
                </div>
              ) : fileContent ? (
                <Editor
                  height="100%"
                  language={fileContent.language}
                  value={fileContent.content}
                  theme={isDark ? 'vs-dark' : 'light'}
                  options={{
                    readOnly: true,
                    minimap: { enabled: false },
                    fontSize: 13,
                    lineNumbers: 'on',
                    scrollBeyondLastLine: false,
                    wordWrap: 'on',
                    automaticLayout: true,
                    padding: { top: 16 },
                  }}
                />
              ) : null}
            </div>

            {/* File Info Sidebar */}
            {fileContent && (fileContent.classes.length > 0 || fileContent.functions.length > 0 || fileContent.docstring) && (
              <div className="w-64 flex-shrink-0 overflow-auto rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
                <div className="p-4">
                  {/* File Description */}
                  {fileContent.docstring && (
                    <div className="mb-4">
                      <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                        <BookOpen className="h-4 w-4" />
                        About
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {fileContent.docstring}
                      </p>
                    </div>
                  )}

                  {/* Classes */}
                  {fileContent.classes.length > 0 && (
                    <div className="mb-4">
                      <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                        <Box className="h-4 w-4 text-purple-500" />
                        Classes ({fileContent.classes.length})
                      </h4>
                      <div className="space-y-2">
                        {fileContent.classes.map((cls) => (
                          <div
                            key={cls.name}
                            className="rounded-lg bg-purple-50 p-2 dark:bg-purple-900/20"
                          >
                            <div className="font-medium text-purple-700 dark:text-purple-400">
                              {cls.name}
                            </div>
                            {cls.docstring && (
                              <p className="mt-1 text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                                {cls.docstring}
                              </p>
                            )}
                            {cls.methods.length > 0 && (
                              <div className="mt-1 text-xs text-purple-600 dark:text-purple-400">
                                {cls.methods.length} methods
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Functions */}
                  {fileContent.functions.length > 0 && (
                    <div>
                      <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                        <Code2 className="h-4 w-4 text-green-500" />
                        Functions ({fileContent.functions.length})
                      </h4>
                      <div className="space-y-2">
                        {fileContent.functions.map((fn) => (
                          <div
                            key={fn.name}
                            className="rounded-lg bg-green-50 p-2 dark:bg-green-900/20"
                          >
                            <div className="font-medium text-green-700 dark:text-green-400">
                              {fn.name}
                            </div>
                            {fn.docstring && (
                              <p className="mt-1 text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                                {fn.docstring}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
