import { useMemo } from 'react'

interface DataPoint {
  iteration: number
  erlang: number
  blocking_prob: number
}

interface ProgressChartProps {
  data: DataPoint[]
  height?: number
}

export function ProgressChart({ data, height = 200 }: ProgressChartProps) {
  const chartData = useMemo(() => {
    if (data.length === 0) return null

    const maxBlocking = Math.max(...data.map((d) => d.blocking_prob), 0.01)
    const width = 100

    // Normalize data points
    const points = data.map((d, i) => ({
      x: (i / Math.max(data.length - 1, 1)) * width,
      y: height - (d.blocking_prob / maxBlocking) * (height - 20) - 10,
      value: d.blocking_prob,
      erlang: d.erlang,
    }))

    // Create path
    const pathD = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
      .join(' ')

    return { points, pathD, maxBlocking }
  }, [data, height])

  if (!chartData || data.length < 2) {
    return (
      <div
        className="flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200"
        style={{ height }}
      >
        <span className="text-sm text-gray-400">
          Waiting for progress data...
        </span>
      </div>
    )
  }

  const latestValue = data[data.length - 1]

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">Blocking Probability</span>
        <span className="text-sm text-gray-500">
          Current: <span className="font-mono font-medium text-fusion-600">
            {(latestValue.blocking_prob * 100).toFixed(4)}%
          </span>
        </span>
      </div>

      <svg
        viewBox={`0 0 100 ${height}`}
        className="w-full"
        style={{ height }}
        preserveAspectRatio="none"
      >
        {/* Grid lines */}
        <g className="text-gray-200">
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
            <line
              key={ratio}
              x1="0"
              y1={10 + ratio * (height - 20)}
              x2="100"
              y2={10 + ratio * (height - 20)}
              stroke="currentColor"
              strokeWidth="0.5"
            />
          ))}
        </g>

        {/* Area fill */}
        <path
          d={`${chartData.pathD} L 100 ${height - 10} L 0 ${height - 10} Z`}
          fill="url(#gradient)"
          opacity="0.3"
        />

        {/* Line */}
        <path
          d={chartData.pathD}
          fill="none"
          stroke="#0284c7"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />

        {/* Latest point */}
        {chartData.points.length > 0 && (
          <circle
            cx={chartData.points[chartData.points.length - 1].x}
            cy={chartData.points[chartData.points.length - 1].y}
            r="3"
            fill="#0284c7"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Gradient definition */}
        <defs>
          <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0284c7" />
            <stop offset="100%" stopColor="#0284c7" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>

      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>Erlang: {data[0]?.erlang?.toFixed(1) ?? '-'}</span>
        <span>Erlang: {latestValue.erlang?.toFixed(1) ?? '-'}</span>
      </div>
    </div>
  )
}
