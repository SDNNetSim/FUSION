import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Network, Info } from 'lucide-react'
import { topologyApi } from '@/api/client'
import { NetworkGraph } from '@/components/topology/NetworkGraph'
import { UtilizationLegend } from '@/components/topology/UtilizationLegend'
import type { TopologyNode, TopologyLink } from '@/api/types'

export function TopologyPage() {
  const [selectedTopology, setSelectedTopology] = useState<string>('')
  const [selectedNode, setSelectedNode] = useState<TopologyNode | null>(null)
  const [selectedLink, setSelectedLink] = useState<TopologyLink | null>(null)

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
  if (!selectedTopology && topologyList?.topologies.length) {
    setSelectedTopology(topologyList.topologies[0].name)
  }

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

        {/* Topology Selector */}
        <div className="flex items-center gap-2">
          <label
            htmlFor="topology-select"
            className="text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Topology:
          </label>
          <select
            id="topology-select"
            value={selectedTopology}
            onChange={(e) => {
              setSelectedTopology(e.target.value)
              setSelectedNode(null)
              setSelectedLink(null)
            }}
            disabled={listLoading}
            className="rounded-md border border-gray-300 px-3 py-1.5 text-sm focus:border-fusion-500 focus:outline-none focus:ring-1 focus:ring-fusion-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
          >
            {listLoading ? (
              <option>Loading...</option>
            ) : (
              topologyList?.topologies.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name} ({t.node_count} nodes, {t.link_count} links)
                </option>
              ))
            )}
          </select>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 gap-4 overflow-hidden">
        {/* Graph */}
        <div className="flex-1 overflow-hidden rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
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
        <div className="w-64 space-y-4">
          {/* Legend */}
          <UtilizationLegend />

          {/* Selected Info */}
          {(selectedNode || selectedLink) && (
            <div className="rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-700 dark:bg-gray-800">
              <h4 className="mb-2 flex items-center gap-2 text-xs font-medium text-gray-700 dark:text-gray-300">
                <Info className="h-3 w-3" />
                Selected
              </h4>

              {selectedNode && (
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Type:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      Node
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">ID:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {selectedNode.id}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Label:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {selectedNode.label}
                    </span>
                  </div>
                </div>
              )}

              {selectedLink && (
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Type:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      Link
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">From:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {selectedLink.source}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">To:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {selectedLink.target}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Length:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {selectedLink.length_km.toFixed(0)} km
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Utilization:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {selectedLink.utilization.toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}

              <button
                onClick={() => {
                  setSelectedNode(null)
                  setSelectedLink(null)
                }}
                className="mt-2 w-full rounded bg-gray-100 px-2 py-1 text-xs text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600"
              >
                Clear selection
              </button>
            </div>
          )}

          {/* Topology Stats */}
          {topology && (
            <div className="rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-700 dark:bg-gray-800">
              <h4 className="mb-2 text-xs font-medium text-gray-700 dark:text-gray-300">
                Statistics
              </h4>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Nodes:</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {topology.nodes.length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Links:</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {topology.links.length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Total Length:</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {topology.links
                      .reduce((sum, l) => sum + l.length_km, 0)
                      .toFixed(0)}{' '}
                    km
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
