import { useState, useMemo, useEffect } from 'react';
import MovieCard from '../components/MovieCard';
import FilterDrawer from '../components/FilterDrawer';
import { fetchCatalog } from '../services/api';

const YEAR_MIN = 1990;
const YEAR_MAX = 2026;
const DEFAULT_FILTERS = {
  genre: 'All', service: 'All', sort: 'chai',
  chaiStatus: 'All', noelStatus: 'All',
  imdbMin: 0, yearMin: YEAR_MIN, yearMax: YEAR_MAX,
};
const CARDS_PER_PAGE = 12;

export default function RecommendPage() {
  const [allMovies, setAllMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [visibleCount, setVisibleCount] = useState(CARDS_PER_PAGE);

  useEffect(() => {
    fetchCatalog()
      .then(data => { setAllMovies(data); setLoading(false); })
      .catch(err => { setError(err.message); setLoading(false); });
  }, []);

  const filtered = useMemo(() => {
    let list = [...allMovies];

    if (filters.genre !== 'All')
      list = list.filter(m => m.genre.includes(filters.genre));
    if (filters.service !== 'All')
      list = list.filter(m => m.service === filters.service);
    if (filters.chaiStatus === 'Seen')
      list = list.filter(m => m.chaiSeen);
    if (filters.chaiStatus === 'Not Seen')
      list = list.filter(m => !m.chaiSeen);
    if (filters.noelStatus === 'Seen')
      list = list.filter(m => m.noelSeen);
    if (filters.noelStatus === 'Not Seen')
      list = list.filter(m => !m.noelSeen);
    if (filters.imdbMin > 0)
      list = list.filter(m => m.imdbScore != null && m.imdbScore >= filters.imdbMin);
    if (filters.yearMin > YEAR_MIN || filters.yearMax < YEAR_MAX)
      list = list.filter(m => m.year >= filters.yearMin && m.year <= filters.yearMax);

    list.sort((a, b) => {
      if (filters.sort === 'chai') return (b.chaiScore ?? 0) - (a.chaiScore ?? 0);
      if (filters.sort === 'noel') return (b.noelScore ?? 0) - (a.noelScore ?? 0);
      if (filters.sort === 'imdb') return (b.imdbScore ?? 0) - (a.imdbScore ?? 0);
      if (filters.sort === 'year') return (b.year ?? 0) - (a.year ?? 0);
      return 0;
    });

    return list;
  }, [allMovies, filters]);

  const visible = filtered.slice(0, visibleCount);
  const hasMore = visibleCount < filtered.length;

  const activeFilterCount = [
    filters.genre !== 'All',
    filters.service !== 'All',
    filters.sort !== 'chai',
    filters.chaiStatus !== 'All',
    filters.noelStatus !== 'All',
    filters.imdbMin > 0,
    filters.yearMin > YEAR_MIN || filters.yearMax < YEAR_MAX,
  ].filter(Boolean).length;

  if (loading) {
    return (
      <div className="page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <span className="material-symbols-outlined" style={{ fontSize: 48, color: '#484847', display: 'block', marginBottom: 12 }}>
            movie_filter
          </span>
          <p style={{ color: '#8b919d', fontSize: 14 }}>Loading catalog…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
        <div style={{ textAlign: 'center', padding: '0 24px' }}>
          <span className="material-symbols-outlined" style={{ fontSize: 48, color: '#484847', display: 'block', marginBottom: 12 }}>
            error
          </span>
          <p style={{ color: '#8b919d', fontSize: 14, marginBottom: 8 }}>Could not connect to Enzyme backend.</p>
          <p style={{ color: '#484847', fontSize: 12 }}>Start the API server: <code style={{ color: '#a4c9ff' }}>python3 api.py</code></p>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      {/* Header */}
      <div style={{
        padding: '20px 16px 12px',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
      }}>
        <div>
          <h1 style={{
            fontFamily: "'Manrope', sans-serif",
            fontSize: 26,
            fontWeight: 800,
            color: '#e5e2e1',
            letterSpacing: '-0.5px',
            lineHeight: 1,
          }}>
            ENZYME
          </h1>
          <p style={{
            fontSize: 12,
            color: '#8b919d',
            fontWeight: 500,
            marginTop: 3,
            letterSpacing: '0.06em',
          }}>
            CINEMATIC CURATOR
          </p>
        </div>

        <button
          onClick={() => setDrawerOpen(true)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '8px 14px',
            background: '#1c1b1b',
            borderRadius: 9999,
            border: '1px solid #484847',
            color: activeFilterCount > 0 ? '#a4c9ff' : '#8b919d',
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 600,
            fontFamily: 'inherit',
            position: 'relative',
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18 }}>tune</span>
          <span>Filter</span>
          {activeFilterCount > 0 && (
            <span style={{
              position: 'absolute',
              top: -4, right: -4,
              width: 16, height: 16,
              background: '#a4c9ff',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 9,
              fontWeight: 700,
              color: '#003258',
            }}>
              {activeFilterCount}
            </span>
          )}
        </button>
      </div>

      {/* Active filter pills */}
      {activeFilterCount > 0 && (
        <div style={{ padding: '0 16px 12px', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {filters.genre !== 'All' && (
            <FilterPill label={filters.genre} onRemove={() => setFilters(f => ({ ...f, genre: 'All' }))} />
          )}
          {filters.service !== 'All' && (
            <FilterPill label={filters.service} onRemove={() => setFilters(f => ({ ...f, service: 'All' }))} />
          )}
          {filters.chaiStatus !== 'All' && (
            <FilterPill label={`Chai: ${filters.chaiStatus}`} onRemove={() => setFilters(f => ({ ...f, chaiStatus: 'All' }))} />
          )}
          {filters.noelStatus !== 'All' && (
            <FilterPill label={`Noel: ${filters.noelStatus}`} color="noel" onRemove={() => setFilters(f => ({ ...f, noelStatus: 'All' }))} />
          )}
          {filters.imdbMin > 0 && (
            <FilterPill label={`IMDb ≥ ${filters.imdbMin.toFixed(1)}`} onRemove={() => setFilters(f => ({ ...f, imdbMin: 0 }))} />
          )}
          {(filters.yearMin > YEAR_MIN || filters.yearMax < YEAR_MAX) && (
            <FilterPill label={`${filters.yearMin}–${filters.yearMax}`} onRemove={() => setFilters(f => ({ ...f, yearMin: YEAR_MIN, yearMax: YEAR_MAX }))} />
          )}
        </div>
      )}

      {/* Sort label */}
      <div style={{ padding: '0 16px 12px', display: 'flex', alignItems: 'center', gap: 6 }}>
        <span className="material-symbols-outlined" style={{ fontSize: 14, color: '#8b919d' }}>sort</span>
        <span style={{ fontSize: 11, color: '#8b919d' }}>
          {filters.sort === 'chai' ? 'Sorted by Chai Score' :
           filters.sort === 'noel' ? 'Sorted by Noel Score' :
           filters.sort === 'imdb' ? 'Sorted by IMDb' : 'Newest First'}
        </span>
        <span style={{ fontSize: 11, color: '#484847', marginLeft: 4 }}>
          {filtered.length} film{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Grid */}
      <div style={{
        padding: '0 12px',
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 12,
      }}>
        {visible.map(movie => (
          <MovieCard key={movie.id} movie={movie} />
        ))}
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <div style={{ padding: '60px 24px', textAlign: 'center' }}>
          <span className="material-symbols-outlined" style={{ fontSize: 48, color: '#484847', display: 'block', marginBottom: 12 }}>
            movie_filter
          </span>
          <p style={{ color: '#8b919d', fontSize: 14 }}>No films match your filters.</p>
          <button
            onClick={() => { setFilters(DEFAULT_FILTERS); }}
            style={{
              marginTop: 16, padding: '8px 20px',
              background: 'rgba(164,201,255,0.12)', color: '#a4c9ff',
              borderRadius: 9999, border: '1px solid rgba(164,201,255,0.3)',
              cursor: 'pointer', fontSize: 13, fontFamily: 'inherit',
            }}
          >
            Clear Filters
          </button>
        </div>
      )}

      {/* Load More */}
      {hasMore && (
        <div style={{ padding: '20px 16px', textAlign: 'center' }}>
          <button
            onClick={() => setVisibleCount(n => n + CARDS_PER_PAGE)}
            style={{
              padding: '12px 32px',
              background: '#1c1b1b',
              color: '#a4c9ff',
              border: '1px solid rgba(164,201,255,0.3)',
              borderRadius: 9999,
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            Load More ({filtered.length - visibleCount} remaining)
          </button>
        </div>
      )}

      <FilterDrawer
        isOpen={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        filters={filters}
        onChange={f => { setFilters(f); setVisibleCount(CARDS_PER_PAGE); }}
        yearMin={YEAR_MIN}
        yearMax={YEAR_MAX}
      />
    </div>
  );
}

function FilterPill({ label, onRemove, color = 'chai' }) {
  const accent = color === 'noel' ? '#ffb4aa' : '#a4c9ff';
  const bg = color === 'noel' ? 'rgba(255,180,170,0.1)' : 'rgba(164,201,255,0.1)';
  const border = color === 'noel' ? 'rgba(255,180,170,0.25)' : 'rgba(164,201,255,0.25)';
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 4,
      padding: '4px 10px 4px 12px', background: bg,
      borderRadius: 9999, border: `1px solid ${border}`,
    }}>
      <span style={{ fontSize: 12, color: accent }}>{label}</span>
      <button onClick={onRemove} style={{ border: 'none', background: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: 0 }}>
        <span className="material-symbols-outlined" style={{ fontSize: 14, color: accent }}>close</span>
      </button>
    </div>
  );
}
