import { useEffect, useState } from 'react'
import { api } from '../api.js'

const NAMES = {
  coarse: 'Sentinel-5P as is (1 km)', raw_1km: 'Sentinel-5P as is (1 km)', bilinear: 'Interpolation', lr: 'Linear regression',
  mslr: 'Multi-scale regression', rf: 'Random forest', cnn: 'CNN',
}
const ORDER = ['coarse', 'raw_1km', 'bilinear', 'lr', 'mslr', 'rf', 'cnn']
const SHORT = { NO2: 'NO₂', SO2: 'SO₂', CO: 'CO', HCHO: 'HCHO', O3: 'O₃', AER_AI: 'Aerosol' }
const num = (v, d = 2) => (v == null || Number.isNaN(v) ? '–' : v.toFixed(d))
const pct = (v) => (v == null || Number.isNaN(v) ? '–' : `${Math.round(v * 100)}%`)

function Table({ rows, cols }) {
  const keys = ORDER.filter((k) => rows[k])
  return (
    <table>
      <thead><tr><th>Method</th>{cols.map((c) => <th key={c.label} title={c.title}>{c.label}</th>)}</tr></thead>
      <tbody>
        {keys.map((k) => (
          <tr key={k}><td>{NAMES[k] || k}</td>
            {cols.map((c) => {
              const vals = keys.map((o) => rows[o][c.key]).filter((v) => v != null && !Number.isNaN(v))
              const best = vals.length > 1 && rows[k][c.key] === (c.lower ? Math.min(...vals) : Math.max(...vals))
              return <td key={c.label} className={best ? 'best' : ''}>{c.fmt(rows[k][c.key])}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function ValidationPanel({ pollutant }) {
  const [v, setV] = useState(null)
  useEffect(() => { api.validation().then(setV).catch(() => {}) }, [])
  if (!v) return null
  const g = v.ground?.[pollutant]
  const u = v.upscaling?.[pollutant]
  const sim = v.simulation
  const air = v.airborne?.NO2
  return (
    <details className="panel" open>
      <summary>How accurate is this? ({SHORT[pollutant] || pollutant})</summary>

      <h4>Against NASA airborne NO₂ maps (New York)</h4>
      {air ? (
        <>
          <p className="hint">
            The only street-scale measurement of the same quantity the satellite sees (the NO₂ column) comes from
            aircraft. None has flown over Mumbai, so the system was run on New York for {air.flights?.length} NASA
            flights in 2018 (LISTOS) and compared with the aircraft maps at 250 m. The CNN is the one trained on
            Mumbai: no New York data was used for training.
            {pollutant !== 'NO2' ? ' This test exists for NO₂ only.' : ''}
          </p>
          <Table rows={air.summary} cols={[
            { key: 'r_250m_median', label: 'r 250 m', title: 'Correlation with the aircraft map at 250 m (median over tiles)', fmt: num },
            { key: 'r_1km_median', label: 'r 1 km', title: 'Correlation at 1 km', fmt: num },
            { key: 'r_detail_sub1km_median', label: 'Detail r', title: 'Correlation of the detail inside each 1 km cell', fmt: num },
            { key: 'r_250m_wins_vs_bilinear', label: 'Beats interp.', title: 'Tiles where the method beats interpolation at 250 m', fmt: (x) => x ?? '–' },
          ]} />
          <p className="hint">
            Median over {air.summary?.cnn?.tiles} tiles of 8 × 8 km. In New York the Mumbai-trained CNN is about as good as
            interpolation at 250 m and slightly better at 1 km: its fine detail does not yet transfer reliably to a
            different city and season. Where the satellite already matches the aircraft well
            ({air.summary_resolved?.cnn?.tiles} tiles), interpolation is hard to beat (r at 250 m:
            {' '}{num(air.summary_resolved?.cnn?.r_250m_median)} CNN vs {num(air.summary_resolved?.bilinear?.r_250m_median)} interpolation).
          </p>
        </>
      ) : <p className="hint">Not run yet. See <code>validation/airborne/airborne.py</code>.</p>}

      <h4>Up-scaling test on real Sentinel-5P (Mumbai)</h4>
      {u ? (
        <>
          <p className="hint">Real 1 km data averaged to {u.k_km} km, downscaled back to 1 km and compared with the real
            1 km values ({u.days} overpasses). "Detail r" scores only the detail inside each {u.k_km} km block.</p>
          <Table rows={u.methods} cols={[
            { key: 'spatial_r_median', label: 'r', title: 'Spatial correlation with real 1 km values', fmt: num },
            { key: 'detail_r_median', label: 'Detail r', title: 'Correlation of the sub-block detail', fmt: num },
            { key: 'nrmse', label: 'NRMSE', title: 'RMSE / mean', fmt: num, lower: true },
            { key: 'skill_vs_bilinear', label: 'Skill', title: '1 − RMSE / RMSE of interpolation', fmt: num },
          ]} />
        </>
      ) : <p className="hint">Not run yet. Run <code>validation/validate.py upscaling</code>.</p>}

      <h4>Against surface monitoring stations (different quantity)</h4>
      <p className="hint">Stations measure the air at street level; the satellite measures the whole column above.
        The two are only loosely related, and neighbouring stations often disagree with each other, so this is a
        consistency check, not an accuracy score.</p>
      {g ? (
        <>
          <p className="hint">
            {g.n_stations} stations · {g.n_days} overpass days · {g.n_pairs} matched readings · {g.source},
            {' '}{g.period?.[0]} to {g.period?.[1]}, 13:00–15:00 IST.
            {g.proxy ? ' The aerosol index is compared with PM2.5 as a proxy.' : ''}
            {' '}µg/m³ values use a calibration fitted on the other stations (leave-one-station-out).
          </p>
          <Table rows={g.predictors} cols={[
            { key: 'daily_spatial_r_median', label: 'Spatial r', title: 'Correlation across stations on the same day (median over days)', fmt: num },
            { key: 'rmse_ug_m3', label: 'RMSE µg/m³', title: 'Calibrated error on stations left out of calibration', fmt: (x) => num(x, 1), lower: true },
            { key: 'skill_vs_raw_1km', label: 'Skill', title: '1 − RMSE / RMSE of the raw 1 km satellite value. Above 0 = better than the satellite as is', fmt: num },
            { key: 'aqi_category_accuracy', label: 'AQI band', title: 'Share of readings in the correct CPCB AQI band', fmt: pct },
            { key: 'hotspot_f1_top_quartile', label: 'Hotspot F1', title: 'Finding the top-quartile stations each day', fmt: num },
          ]} />
        </>
      ) : (
        <p className="hint">Not run yet for this pollutant. Run <code>validation/validate.py ground</code> (see docs/VALIDATION.md).
          {pollutant === 'HCHO' ? ' Formaldehyde is not monitored at ground stations.' : ''}</p>
      )}

      {sim?.[pollutant] && (
        <>
          <h4>Simulated scene with a known 10 m field</h4>
          <p className="hint">Tests whether a method can recover 10 m detail at all. It's a controlled test, not Mumbai's real air.</p>
          <Table rows={sim[pollutant]} cols={[
            { key: 'hotspot_class_accuracy', label: 'Class accuracy', title: 'Pixels in the right class: low / moderate / high / hotspot', fmt: pct },
            { key: 'hotspot_f1', label: 'Hotspot F1', fmt: num },
            { key: 'r2', label: 'R²', fmt: num },
          ]} />
        </>
      )}
    </details>
  )
}
