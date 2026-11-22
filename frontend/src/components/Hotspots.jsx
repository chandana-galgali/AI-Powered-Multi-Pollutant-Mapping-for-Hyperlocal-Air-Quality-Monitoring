export default function Hotspots({ hotspots, unit, onPick }) {
  if (!hotspots?.length) return null
  return (
    <section className="panel">
      <h3>Highest readings</h3>
      <ol className="hotspots">
        {hotspots.map((h) => (
          <li key={h.rank}>
            <button onClick={() => onPick({ ...h, t: Date.now() })}>
              <span>{h.lat.toFixed(4)}° N, {h.lon.toFixed(4)}° E</span>
              <b>{h.value.toExponential(2)}</b>
            </button>
          </li>
        ))}
      </ol>
      <p className="hint">{unit}. Select one to zoom in.</p>
    </section>
  )
}
