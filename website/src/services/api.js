const API_BASE = 'http://localhost:5001';

export async function fetchCatalog() {
  const res = await fetch(`${API_BASE}/api/catalog`);
  if (!res.ok) throw new Error('Failed to fetch catalog');
  return res.json();
}

export async function fetchMovie(id) {
  const res = await fetch(`${API_BASE}/api/movie/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error('Movie not found');
  return res.json();
}
