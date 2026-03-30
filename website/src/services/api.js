const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5001';

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

export async function fetchMovieML(id) {
  const res = await fetch(`${API_BASE}/api/movie/${encodeURIComponent(id)}/ml`);
  if (!res.ok) throw new Error('ML fetch failed');
  return res.json();
}

export async function searchMovies(q) {
  const res = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(q)}`);
  if (!res.ok) throw new Error('Search failed');
  return res.json();
}
