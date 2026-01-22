const legendItems = [
  { label: 'Idle', color: '#94a3b8', range: '0%' },
  { label: 'Low', color: '#22c55e', range: '1-24%' },
  { label: 'Medium', color: '#84cc16', range: '25-49%' },
  { label: 'High', color: '#eab308', range: '50-74%' },
  { label: 'Very High', color: '#f97316', range: '75-89%' },
  { label: 'Critical', color: '#ef4444', range: '90-100%' },
]

export function UtilizationLegend() {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-700 dark:bg-gray-800">
      <h4 className="mb-2 text-xs font-medium text-gray-700 dark:text-gray-300">
        Link Utilization
      </h4>
      <div className="space-y-1">
        {legendItems.map((item) => (
          <div key={item.label} className="flex items-center gap-2 text-xs">
            <div
              className="h-3 w-6 rounded"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-gray-600 dark:text-gray-400">{item.label}</span>
            <span className="ml-auto text-gray-400 dark:text-gray-500">
              {item.range}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
