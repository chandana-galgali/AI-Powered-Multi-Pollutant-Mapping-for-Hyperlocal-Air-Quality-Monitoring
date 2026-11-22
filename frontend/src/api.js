const BASE = import.meta.env.VITE_API_BASE || ''

async function req(path, opts) {
  const r = await fetch(BASE + path, opts)
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(body.error || `Request failed (HTTP ${r.status})`)
  return body
}

export const fileUrl = (jobId, name) => `${BASE}/api/jobs/${jobId}/${name}`

export const api = {
  health: () => req('/api/health'),
  pollutants: () => req('/api/pollutants'),
  methods: () => req('/api/methods'),
  areas: () => req('/api/areas'),
  validation: () => req('/api/validation'),
  job: (id) => req(`/api/jobs/${id}`),
  submit: (params) =>
    req('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
}
