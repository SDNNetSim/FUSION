import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import { RunListPage } from '@/pages/RunListPage'
import { NewRunPage } from '@/pages/NewRunPage'
import { RunDetailPage } from '@/pages/RunDetailPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<RunListPage />} />
        <Route path="runs" element={<RunListPage />} />
        <Route path="runs/new" element={<NewRunPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
      </Route>
    </Routes>
  )
}

export default App
