import { useState, useRef, useCallback, useMemo, useEffect } from 'react'
import type { TopologyNode, TopologyLink } from '@/api/types'

interface NetworkGraphProps {
  nodes: TopologyNode[]
  links: TopologyLink[]
  onNodeClick?: (node: TopologyNode) => void
  onLinkClick?: (link: TopologyLink) => void
}

interface TooltipState {
  visible: boolean
  x: number
  y: number
  content: string
}

// Color scale for utilization (0-100%)
function getUtilizationColor(utilization: number): string {
  if (utilization <= 0) return '#94a3b8' // gray-400
  if (utilization < 25) return '#22c55e' // green-500
  if (utilization < 50) return '#84cc16' // lime-500
  if (utilization < 75) return '#eab308' // yellow-500
  if (utilization < 90) return '#f97316' // orange-500
  return '#ef4444' // red-500
}

export function NetworkGraph({
  nodes,
  links,
  onNodeClick,
  onLinkClick,
}: NetworkGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    x: 0,
    y: 0,
    content: '',
  })
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 })
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState({ x: 0, y: 0 })

  // Track container size for responsiveness
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateDimensions = () => {
      setDimensions({
        width: container.clientWidth,
        height: container.clientHeight,
      })
    }

    updateDimensions()

    const resizeObserver = new ResizeObserver(updateDimensions)
    resizeObserver.observe(container)

    return () => resizeObserver.disconnect()
  }, [])

  // Create a map for quick node lookup
  const nodeMap = useMemo(() => {
    const map = new Map<string, TopologyNode>()
    nodes.forEach((node) => map.set(node.id, node))
    return map
  }, [nodes])

  // Calculate bounds and center the graph
  const viewBox = useMemo(() => {
    if (nodes.length === 0) return { minX: 0, minY: 0, width: 800, height: 600 }

    const xs = nodes.map((n) => n.x)
    const ys = nodes.map((n) => n.y)
    const minX = Math.min(...xs) - 50
    const maxX = Math.max(...xs) + 50
    const minY = Math.min(...ys) - 50
    const maxY = Math.max(...ys) + 50

    return {
      minX,
      minY,
      width: maxX - minX,
      height: maxY - minY,
    }
  }, [nodes])

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button === 0) {
        setIsPanning(true)
        setPanStart({ x: e.clientX - transform.x, y: e.clientY - transform.y })
      }
    },
    [transform]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (isPanning) {
        setTransform((prev) => ({
          ...prev,
          x: e.clientX - panStart.x,
          y: e.clientY - panStart.y,
        }))
      }
    },
    [isPanning, panStart]
  )

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
  }, [])

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setTransform((prev) => ({
      ...prev,
      scale: Math.min(Math.max(prev.scale * delta, 0.1), 5),
    }))
  }, [])

  const showTooltip = useCallback((e: React.MouseEvent, content: string) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (rect) {
      setTooltip({
        visible: true,
        x: e.clientX - rect.left + 10,
        y: e.clientY - rect.top - 10,
        content,
      })
    }
  }, [])

  const hideTooltip = useCallback(() => {
    setTooltip((prev) => ({ ...prev, visible: false }))
  }, [])

  const resetView = useCallback(() => {
    setTransform({ x: 0, y: 0, scale: 1 })
  }, [])

  return (
    <div ref={containerRef} className="relative h-full w-full">
      {/* Controls */}
      <div className="absolute right-2 top-2 z-10 flex gap-1">
        <button
          onClick={() => setTransform((p) => ({ ...p, scale: p.scale * 1.2 }))}
          className="rounded bg-white px-2 py-1 text-sm shadow hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600"
          title="Zoom in"
        >
          +
        </button>
        <button
          onClick={() => setTransform((p) => ({ ...p, scale: p.scale * 0.8 }))}
          className="rounded bg-white px-2 py-1 text-sm shadow hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600"
          title="Zoom out"
        >
          -
        </button>
        <button
          onClick={resetView}
          className="rounded bg-white px-2 py-1 text-sm shadow hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600"
          title="Reset view"
        >
          Reset
        </button>
      </div>

      {/* Tooltip */}
      {tooltip.visible && (
        <div
          className="pointer-events-none absolute z-20 rounded bg-gray-900 px-2 py-1 text-xs text-white shadow-lg dark:bg-gray-700"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          {tooltip.content}
        </div>
      )}

      {/* SVG Canvas */}
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        viewBox={`${viewBox.minX} ${viewBox.minY} ${viewBox.width} ${viewBox.height}`}
        className="cursor-grab bg-white active:cursor-grabbing dark:bg-gray-800"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <g
          transform={`translate(${transform.x / transform.scale}, ${transform.y / transform.scale}) scale(${transform.scale})`}
        >
          {/* Links */}
          {links.map((link) => {
            const source = nodeMap.get(link.source)
            const target = nodeMap.get(link.target)
            if (!source || !target) return null

            return (
              <line
                key={link.id}
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke={getUtilizationColor(link.utilization)}
                strokeWidth={3}
                className="cursor-pointer transition-all hover:stroke-[5]"
                onMouseEnter={(e) =>
                  showTooltip(
                    e,
                    `${link.source} - ${link.target}\nLength: ${link.length_km.toFixed(0)} km\nUtilization: ${link.utilization.toFixed(1)}%`
                  )
                }
                onMouseLeave={hideTooltip}
                onClick={() => onLinkClick?.(link)}
              />
            )
          })}

          {/* Nodes */}
          {nodes.map((node) => (
            <g
              key={node.id}
              transform={`translate(${node.x}, ${node.y})`}
              className="cursor-pointer"
              onMouseEnter={(e) => showTooltip(e, `Node ${node.id}\n${node.label}`)}
              onMouseLeave={hideTooltip}
              onClick={() => onNodeClick?.(node)}
            >
              <circle
                r={12}
                fill="#3b82f6"
                stroke="#1d4ed8"
                strokeWidth={2}
                className="transition-all hover:fill-blue-400"
              />
              <text
                y={25}
                textAnchor="middle"
                className="fill-gray-700 text-xs font-medium dark:fill-gray-300"
                style={{ fontSize: '10px' }}
              >
                {node.id}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  )
}
