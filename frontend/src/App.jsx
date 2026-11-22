import { useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import MapView from './components/MapView.jsx'
import Controls from './components/Controls.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import Hotspots from './components/Hotspots.jsx'
import ValidationPanel from './components/ValidationPanel.jsx'

const daysAgo = (n) => new Date(Date.now() - n * 864e5).toISOString().slice(0, 10)

export default function App() {
  const [catalog, setCatalog] = useState({ pollutants: {}, methods: {}, areas: {}, limits: {} })
  const [health, setHealth] = useState(null)
  const [params, setParams] = useState({ pollutant: 'NO2', method: 'mslr', date: daysAgo(5), area: 'chembur_trombay' })
  const [viewBbox, setViewBbox] = useState(null)
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const [layer, setLayer] = useState('fine')
  const [opacity, setOpacity] = useState(0.8)
  const [focus, setFocus] = useState(null)
  const poll = useRef(null)

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(`Can't reach the server: ${e.message}`))
    Promise.all([api.pollutants(), api.methods(), api.areas()])
      .then(([pollutants, methods, a]) => setCatalog({ pollutants, methods, areas: a.areas, limits: a.max_area_km2 }))
      .catch(() => {})
    return () => clearTimeout(poll.current)
  }, [])

  const track = (id) => {
    api.job(id).then((j) => {
      setJob(j)
      if (j.state === 'queued' || j.state === 'running') poll.current = setTimeout(() => track(id), 1500)
      if (j.state === 'error') setError(j.error)
    }).catch((e) => setError(e.message))
  }

  const run = async () => {
    clearTimeout(poll.current)
    setError(null)
    const body = { ...params }
    if (params.area === 'custom') {
      if (!viewBbox) return setError('Move the map to the area you want first.')
      body.bbox = viewBbox
    }
    try {
      const j = await api.submit(body)
      setJob(j)
      setLayer('fine')
      track(j.id)
    } catch (e) {
      setError(e.message)
    }
  }

  const busy = job && (job.state === 'queued' || job.state === 'running')
  const result = job?.state === 'done' ? job.result : null
  const selectedArea = params.area !== 'custom' ? catalog.areas[params.area] : null

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>Hyperlocal air quality, Mumbai</h1>
          <p className="lede">
            Sentinel-5P measures pollution in 1 km blocks. This system learns how each pollutant relates to
            Sentinel-2's 10 m bands (vegetation, water, built-up surfaces) and redraws every block at 10 m,
            staying consistent with what the satellite measured.
          </p>
        </header>
        {health && !health.earth_engine?.connected && (
          <div className="error">Earth Engine isn't connected on this server, so maps can't be produced.
            {health.earth_engine?.error ? ` (${health.earth_engine.error})` : ''}</div>
        )}
        <Controls params={params} setParams={setParams} catalog={catalog} onRun={run} busy={busy} />
        {busy && (
          <div className="progress" role="status">
            <div className="bar"><i style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} /></div>
            <span>{job.stage}…</span>
          </div>
        )}
        {error && <div className="error">{error}</div>}
        {result && (
          <>
            <div className="toggle" role="group" aria-label="Map layer">
              <button aria-pressed={layer === 'coarse'} onClick={() => setLayer('coarse')}>Sentinel-5P, 1 km</button>
              <button aria-pressed={layer === 'fine'} onClick={() => setLayer('fine')}>Downscaled, {result.res_m} m</button>
            </div>
            <label className="row">Opacity
              <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
            </label>
            <ResultPanel job={job} result={result} />
            <Hotspots hotspots={result.hotspots} unit={result.unit} onPick={setFocus} />
          </>
        )}
        <ValidationPanel pollutant={params.pollutant} />
      </aside>
      <main className="map">
        <MapView result={result} layer={layer} opacity={opacity} focus={focus}
          areaBbox={selectedArea?.bbox} onView={setViewBbox} />
      </main>
    </div>
  )
}
