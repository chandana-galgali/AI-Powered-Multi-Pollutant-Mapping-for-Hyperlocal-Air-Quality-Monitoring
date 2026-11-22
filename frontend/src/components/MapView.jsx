import { useEffect } from 'react'
import { MapContainer, TileLayer, ImageOverlay, CircleMarker, Tooltip, Rectangle, LayersControl, useMap, useMapEvents } from 'react-leaflet'

const toLeaflet = (b) => [[b[1], b[0]], [b[3], b[2]]]

function Fit({ bounds }) {
  const map = useMap()
  useEffect(() => { if (bounds) map.fitBounds(toLeaflet(bounds), { padding: [20, 20] }) }, [bounds?.join(',')]) // eslint-disable-line
  return null
}

function Focus({ focus }) {
  const map = useMap()
  useEffect(() => { if (focus) map.flyTo([focus.lat, focus.lon], 16) }, [focus]) // eslint-disable-line
  return null
}

function ViewReporter({ onView }) {
  const map = useMapEvents({
    moveend: () => {
      const b = map.getBounds()
      onView([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()])
    },
  })
  useEffect(() => {
    const b = map.getBounds()
    onView([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()])
  }, []) // eslint-disable-line
  return null
}

export default function MapView({ result, layer, opacity, focus, areaBbox, onView }) {
  return (
    <MapContainer center={[19.06, 72.88]} zoom={12} style={{ height: '100%', width: '100%' }}>
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="Satellite imagery">
          <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution="Imagery © Esri" maxZoom={19} />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Streets">
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="© OpenStreetMap contributors" maxZoom={19} />
        </LayersControl.BaseLayer>
      </LayersControl>
      {areaBbox && !result && (
        <Rectangle bounds={toLeaflet(areaBbox)} pathOptions={{ color: '#E0A526', weight: 2, fill: false, dashArray: '6 4' }} />
      )}
      {result && (
        <ImageOverlay key={`${result.params.pollutant}-${result.params.method}-${layer}-${result.bounds.join()}`}
          url={layer === 'fine' ? result.overlay : result.coarse_overlay}
          bounds={toLeaflet(result.bounds)} opacity={opacity}
          className={layer === 'coarse' ? 'pixelated' : ''} />
      )}
      {result?.hotspots?.map((h) => (
        <CircleMarker key={h.rank} center={[h.lat, h.lon]} radius={7}
          pathOptions={{ color: '#fff', weight: 2, fillColor: '#6e1240', fillOpacity: 0.95 }}>
          <Tooltip>Hotspot {h.rank}: {h.value.toExponential(2)} {result.unit}</Tooltip>
        </CircleMarker>
      ))}
      <Fit bounds={result?.bounds || areaBbox} />
      <Focus focus={focus} />
      <ViewReporter onView={onView} />
    </MapContainer>
  )
}
