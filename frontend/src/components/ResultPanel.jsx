import { fileUrl } from '../api.js'

const sci = (v) => (v == null ? '–' : Math.abs(v) >= 0.01 && Math.abs(v) < 1000 ? v.toFixed(3) : v.toExponential(2))
const pct = (v) => `${Math.round(v * 100)}%`

export default function ResultPanel({ job, result }) {
  const { legend, stats, provenance: pv, params } = result
  const grad = legend.palette.map((c) => `rgb(${c.join(',')})`).join(',')
  return (
    <section className="panel">
      <h3>{result.label}</h3>
      <div className="legend">
        <div className="ramp" style={{ background: `linear-gradient(90deg, ${grad})` }} />
        <div className="ticks"><span>{sci(legend.min)}</span><span>{result.unit}</span><span>{sci(legend.max)}</span></div>
      </div>
      <table>
        <tbody>
          <tr><td>Area mean</td><td>{sci(stats.mean)}</td></tr>
          <tr><td>95th percentile</td><td>{sci(stats.p95)}</td></tr>
          <tr><td>Highest pixel</td><td>{sci(stats.max)}</td></tr>
          <tr><td>Satellite blocks under cloud</td><td>{pct(stats.cloud_fraction)}</td></tr>
          <tr><td>Match with satellite at 1 km (RMSE)</td><td>{sci(stats.satellite_consistency_rmse)}</td></tr>
        </tbody>
      </table>
      <h4>Data used</h4>
      <ul className="prov">
        <li>Method: {result.method_label}{pv.cnn ? ` (${pv.cnn})` : ''}</li>
        <li>Sentinel-5P overpass shown: {pv.target_day}; model fitted on {stats.overpasses_used} overpasses
          ({pv.s5p_days[0]} to {pv.s5p_days[pv.s5p_days.length - 1]})</li>
        <li>{pv.s2}</li>
        <li>Output: {result.size[0]} × {result.size[1]} pixels at {result.res_m} m</li>
      </ul>
      <div className="downloads">
        <a className="btn" href={fileUrl(job.id, 'geotiff')}>Download {result.res_m} m GeoTIFF</a>
        <a className="btn ghost" href={fileUrl(job.id, 'coarse_geotiff')}>1 km input GeoTIFF</a>
      </div>
      <p className="hint">Finished in {result.runtime_s} s · {params.pollutant}, {params.date}</p>
    </section>
  )
}
