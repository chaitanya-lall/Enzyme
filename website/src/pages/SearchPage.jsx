import { useState } from 'react';

export default function SearchPage() {
  const [query, setQuery] = useState('');

  return (
    <div className="page" style={{ padding: '20px 16px' }}>
      <h1 style={{
        fontFamily: "'Manrope', sans-serif",
        fontSize: 26,
        fontWeight: 800,
        color: '#e5e2e1',
        letterSpacing: '-0.5px',
        lineHeight: 1,
        marginBottom: 3,
      }}>
        ENZYME
      </h1>
      <p style={{
        fontSize: 12,
        color: '#8b919d',
        fontWeight: 500,
        letterSpacing: '0.06em',
        marginBottom: 20,
      }}>
        CINEMATIC CURATOR
      </p>

      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        background: '#1c1b1b', borderRadius: 12,
        border: '1px solid #353534', padding: '10px 14px',
      }}>
        <span className="material-symbols-outlined" style={{ fontSize: 18, color: '#8b919d', flexShrink: 0 }}>search</span>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search films, directors…"
          autoFocus
          style={{
            flex: 1, background: 'none', border: 'none', outline: 'none',
            color: '#e5e2e1', fontSize: 14, fontFamily: 'inherit',
          }}
        />
        {query && (
          <button onClick={() => setQuery('')} style={{ border: 'none', background: 'none', cursor: 'pointer', display: 'flex', padding: 0 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 16, color: '#8b919d' }}>close</span>
          </button>
        )}
      </div>
    </div>
  );
}
