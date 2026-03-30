import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { searchMovies } from '../services/api';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Debounced search — fires 400ms after user stops typing
  useEffect(() => {
    if (!query.trim()) { setResults([]); return; }
    const timer = setTimeout(() => {
      setLoading(true);
      searchMovies(query)
        .then(data => { setResults(data); setLoading(false); })
        .catch(() => setLoading(false));
    }, 400);
    return () => clearTimeout(timer);
  }, [query]);

  return (
    <div className="page" style={{ padding: '20px 16px' }}>
      <h1 style={{
        fontFamily: "'Manrope', sans-serif",
        fontSize: 26, fontWeight: 800, color: '#e5e2e1',
        letterSpacing: '-0.5px', lineHeight: 1, marginBottom: 3,
      }}>
        ENZYME
      </h1>
      <p style={{
        fontSize: 12, color: '#8b919d', fontWeight: 500,
        letterSpacing: '0.06em', marginBottom: 20,
      }}>
        CINEMATIC CURATOR
      </p>

      {/* Search input */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        background: '#1c1b1b', borderRadius: 12,
        border: '1px solid #353534', padding: '10px 14px',
        marginBottom: 16,
      }}>
        <span className="material-symbols-outlined" style={{ fontSize: 18, color: '#8b919d', flexShrink: 0 }}>search</span>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search any film…"
          autoFocus
          style={{
            flex: 1, background: 'none', border: 'none', outline: 'none',
            color: '#e5e2e1', fontSize: 14, fontFamily: 'inherit',
          }}
        />
        {loading && (
          <span className="material-symbols-outlined" style={{ fontSize: 16, color: '#8b919d', animation: 'spin 1s linear infinite' }}>progress_activity</span>
        )}
        {query && !loading && (
          <button onClick={() => { setQuery(''); setResults([]); }} style={{ border: 'none', background: 'none', cursor: 'pointer', display: 'flex', padding: 0 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 16, color: '#8b919d' }}>close</span>
          </button>
        )}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {results.map(movie => (
            <SearchRow key={movie.id} movie={movie} onClick={() => navigate(`/detail/${movie.id}`)} />
          ))}
        </div>
      )}

      {query.trim() && !loading && results.length === 0 && (
        <p style={{ color: '#484847', fontSize: 13, textAlign: 'center', marginTop: 40 }}>
          No films found for "{query}"
        </p>
      )}
    </div>
  );
}

function SearchRow({ movie, onClick }) {
  const chaiColor = movie.chaiScore >= 85 ? '#a4c9ff' : movie.chaiScore >= 70 ? '#b1c8ed' : '#8b919d';
  const noelColor = movie.noelScore >= 85 ? '#ffb4aa' : movie.noelScore >= 70 ? '#b1c8ed' : '#8b919d';

  return (
    <div
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '10px 12px', borderRadius: 12,
        cursor: 'pointer', background: 'transparent',
        transition: 'background 0.15s',
      }}
      onMouseEnter={e => e.currentTarget.style.background = '#1c1b1b'}
      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
    >
      <img
        src={movie.poster}
        alt={movie.title}
        style={{ width: 44, height: 66, objectFit: 'cover', borderRadius: 6, flexShrink: 0, background: '#262625' }}
        onError={e => { e.target.style.display = 'none'; }}
      />
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ fontSize: 14, fontWeight: 600, color: '#e5e2e1', marginBottom: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {movie.title}
        </p>
        <p style={{ fontSize: 12, color: '#8b919d' }}>{movie.year}</p>
      </div>
      <div style={{ display: 'flex', gap: 10, flexShrink: 0 }}>
        {movie.chaiScore != null ? (
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: 14, fontWeight: 700, color: chaiColor, lineHeight: 1 }}>{movie.chaiScore}%</p>
            <p style={{ fontSize: 9, color: '#8b919d', letterSpacing: '0.04em' }}>CHAI</p>
          </div>
        ) : (
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: 14, fontWeight: 700, color: '#484847', lineHeight: 1 }}>—</p>
            <p style={{ fontSize: 9, color: '#484847', letterSpacing: '0.04em' }}>CHAI</p>
          </div>
        )}
        {movie.noelScore != null ? (
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: 14, fontWeight: 700, color: noelColor, lineHeight: 1 }}>{movie.noelScore}%</p>
            <p style={{ fontSize: 9, color: '#8b919d', letterSpacing: '0.04em' }}>NOEL</p>
          </div>
        ) : (
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: 14, fontWeight: 700, color: '#484847', lineHeight: 1 }}>—</p>
            <p style={{ fontSize: 9, color: '#484847', letterSpacing: '0.04em' }}>NOEL</p>
          </div>
        )}
      </div>
    </div>
  );
}
