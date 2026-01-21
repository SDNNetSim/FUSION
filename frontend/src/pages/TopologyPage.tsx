import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Network, Info, ChevronDown, Check } from 'lucide-react'
import { topologyApi } from '@/api/client'
import { NetworkGraph } from '@/components/topology/NetworkGraph'
import { UtilizationLegend } from '@/components/topology/UtilizationLegend'
import type { TopologyNode, TopologyLink } from '@/api/types'

export function TopologyPage() {
  const [selectedTopology, setSelectedTopology] = useState<string>('')
  const [selectedNode, setSelectedNode] = useState<TopologyNode | null>(null)
  const [selectedLink, setSelectedLink] = useState<TopologyLink | null>(null)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const { data: topologyList, isLoading: listLoading } = useQuery({
    queryKey: ['topologies'],
    queryFn: topologyApi.list,
  })

  const { data: topology, isLoading: topologyLoading } = useQuery({
    queryKey: ['topology', selectedTopology],
    queryFn: () => topologyApi.get(selectedTopology),
    enabled: !!selectedTopology,
  })

  // Auto-select first topology
  useEffect(() => {
    if (!selectedTopology && topologyList?.topologies.length) {
      setSelectedTopology(topologyList.topologies[0].name)
    }
  }, [selectedTopology, topologyList])

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

  const selectedTopologyInfo = topologyList?.topologies.find(t => t.name === selectedTopology)

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Network Topology
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Visualize network structure and link utilization
          </p>
        </div>

        {/* Custom Topology Selector */}
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            disabled={listLoading}
            className="flex items-center gap-3 rounded-lg border border-gray-200 bg-white px-4 py-2.5 text-left shadow-sm transition-all hover:border-gray-300 focus:border-fusion-500 focus:outline-none focus:ring-2 focus:ring-fusion-500/20 dark:border-gray-600 dark:bg-gray-800 dark:hover:border-gray-500"
          >
            <Network className="h-5 w-5 text-fusion-500" />
            <div className="min-w-[140px]">
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {listLoading ? 'Loading...' : selectedTopology || 'Select topology'}
              </div>
              {selectedTopologyInfo && (
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {selectedTopologyInfo.node_count} nodes, {selectedTopologyInfo.link_count} links
                </div>
              )}
            </div>
            <ChevronDown className={`h-4 w-4 text-gray-400 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
          </button>

          {/* Dropdown Menu */}
          {dropdownOpen && topologyList && (
            <div className="absolute right-0 top-full z-50 mt-2 w-72 rounded-lg border border-gray-200 bg-white py-1 shadow-lg dark:border-gray-700 dark:bg-gray-800">
              {topologyList.topologies.map((t) => (
                <button
                  key={t.name}
                  onClick={() => {
                    setSelectedTopology(t.name)
                    setSelectedNode(null)
                    setSelectedLink(null)
                    setDropdownOpen(false)
                  }}
                  className={`flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                    selectedTopology === t.name
                      ? 'bg-fusion-50 dark:bg-fusion-900/20'
                      : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                >
                  <div className="flex-1">
                    <div className={`text-sm font-medium ${
                      selectedTopology === t.name
                        ? 'text-fusion-700 dark:text-fusion-400'
                        : 'text-gray-900 dark:text-gray-100'
                    }`}>
                      {t.name}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {t.node_count} nodes, {t.link_count} links
                    </div>
                  </div>
                  {selectedTopology === t.name && (
                    <Check className="h-4 w-4 text-fusion-600" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 gap-4 min-h-0">
        {/* Graph */}
        <div className="flex-1 min-w-0 overflow-hidden rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
          {topologyLoading ? (
            <div className="flex h-full items-center justify-center">
              <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
                <Network className="h-5 w-5 animate-pulse" />
                Loading topology...
              </div>
            </div>
          ) : topology ? (
            <NetworkGraph
              nodes={topology.nodes}
              links={topology.links}
              width={800}
              height={550}
              onNodeClick={setSelectedNode}
              onLinkClick={setSelectedLink}
            />
          ) : (
            <div className="flex h-full items-center justify-center">
              <p className="text-gray-500 dark:text-gray-400">
                Select a topology to view
              </p>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="w-56 flex-shrink-0 space-y-4 overflow-y-auto">
          {/* Legend */}
          <UtilizationLegend />

          {/* Selected Info */}
          {(selectedNode || selectedLink) && (
            <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
              <h4 className="mb-3 flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                <Info className="h-4 w-4" />
                Selected Element
              </h4>

              {selectedNode && (
                <div className="space-y-2">
                  <div className="rounded-lg bg-blue-50 p-3 dark:bg-blue-900/20">
                    <div className="text-xs font-medium uppercase text-blue-600 dark:text-blue-400">Node</div>
                    <div className="mt-1 text-lg font-bold text-gray-900 dark:text-gray-100">{selectedNode.id}</div>
                    <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">{selectedNode.label}</div>
                  </div>
                </div>
              )}

              {selectedLink && (
                <div className="space-y-2">
                  <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-700">
                    <div className="text-xs font-medium uppercase text-gray-500 dark:text-gray-400">Link</div>
                    <div className="mt-1 text-lg font-bold text-gray-900 dark:text-gray-100">
                      {selectedLink.source} - {selectedLink.target}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded-lg bg-gray-50 p-2 dark:bg-gray-700">
                      <div className="text-xs text-gray-500 dark:text-gray-400">Length</div>
                      <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {selectedLink.length_km.toFixed(0)} km
                      </div>
                    </div>
                    <div className="rounded-lg bg-gray-50 p-2 dark:bg-gray-700">
                      <div className="text-xs text-gray-500 dark:text-gray-400">Utilization</div>
                      <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {selectedLink.utilization.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <button
                onClick={() => {
                  setSelectedNode(null)
                  setSelectedLink(null)
                }}
                className="mt-3 w-full rounded-lg bg-gray-100 px-3 py-2 text-sm font-medium text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600"
              >
                Clear selection
              </button>
            </div>
          )}

          {/* Topology Stats */}
          {topology && (
            <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
              <h4 className="mb-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                Network Statistics
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Nodes</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                    {topology.nodes.length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Links</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                    {topology.links.length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Total length</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                    {topology.links.reduce((sum, l) => sum + l.length_km, 0).toLocaleString()} km
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Avg degree</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                    {((topology.links.length * 2) / topology.nodes.length).toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
