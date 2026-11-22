const SHORT = { NO2: 'NO₂', SO2: 'SO₂', CO: 'CO', HCHO: 'HCHO', O3: 'O₃', AER_AI: 'Aerosol' }
const POL_ORDER = ['NO2', 'SO2', 'CO', 'HCHO', 'O3', 'AER_AI']
const METHOD_ORDER = ['bilinear', 'lr', 'mslr', 'rf', 'cnn']
const ordered = (obj, order) => order.filter((k) => obj[k]).map((k) => [k, obj[k]])
const today = () => new Date().toISOString().slice(0, 10)

export default function Controls({ params, setParams, catalog, onRun, busy }) {
  const set = (k, v) => setParams({ ...params, [k]: v })
  const pol = catalog.pollutants[params.pollutant]
  const cnn = catalog.methods.cnn
  const cnnReady = cnn?.pretrained_for?.includes(params.pollutant)
  const isCity = params.area === 'mumbai'
  return (
    <div className="controls">
      <fieldset>
        <legend>Pollutant</legend>
        <div className="chips">
          {ordered(catalog.pollutants, POL_ORDER).map(([k]) => (
            <button key={k} className="chip" aria-pressed={params.pollutant === k}
              title={catalog.pollutants[k].label} onClick={() => set('pollutant', k)}>{SHORT[k] || k}</button>
          ))}
        </div>
        {pol && <p className="hint">{pol.label} · Sentinel-5P {pol.collection.split('/').pop()}</p>}
      </fieldset>

      <label>Date
        <input type="date" value={params.date} min={pol?.start_date} max={today()}
          onChange={(e) => set('date', e.target.value)} />
        <span className="hint">The map uses the last Sentinel-5P overpass on or before this date. Processed data
          usually appears 2–5 days after the overpass.</span>
      </label>

      <label>Area
        <select value={params.area} onChange={(e) => set('area', e.target.value)}>
          {ordered(catalog.areas, ['chembur_trombay', 'somaiya', 'bkc_dharavi', 'andheri_airport', 'western_suburbs', 'south_mumbai', 'mumbai']).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
          <option value="custom">Current map view (up to {catalog.limits?.[10] ?? 160} km² at 10 m)</option>
        </select>
      </label>

      <label>Method
        <select value={params.method} onChange={(e) => set('method', e.target.value)}>
          {ordered(catalog.methods, METHOD_ORDER).map(([k, v]) => (
            <option key={k} value={k} disabled={k === 'cnn' && isCity}>
              {v.label}{k === 'cnn' ? (cnnReady ? ' · pretrained' : ' · trains on request, slower') : ''}
            </option>
          ))}
        </select>
      </label>

      <button className="run" onClick={onRun} disabled={busy}>{busy ? 'Working…' : 'Make the map'}</button>
    </div>
  )
}
