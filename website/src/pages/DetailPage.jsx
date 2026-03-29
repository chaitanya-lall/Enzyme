import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ChromeTab from '../components/ChromeTab';
import OverviewTab from '../components/tabs/OverviewTab';
import ChaiTab from '../components/tabs/ChaiTab';
import NoelTab from '../components/tabs/NoelTab';
import { fetchMovie } from '../services/api';

const TABS = [
  { key: 'overview', label: 'Overview' },
  { key: 'chai', label: 'Chai' },
  { key: 'noel', label: 'Noel' },
];

export default function DetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [movie, setMovie] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    setLoading(true);
    fetchMovie(id)
      .then(data => { setMovie(data); setLoading(false); })
      .catch(() => { setMovie(null); setLoading(false); });
  }, [id]);

  if (loading) {
    return (
      <div className="page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <span className="material-symbols-outlined" style={{ fontSize: 48, color: '#484847', display: 'block', marginBottom: 12 }}>
            movie
          </span>
          <p style={{ color: '#8b919d', fontSize: 14 }}>Loading…</p>
        </div>
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <span className="material-symbols-outlined" style={{ fontSize: 48, color: '#484847', display: 'block', marginBottom: 12 }}>
            movie_off
          </span>
          <p style={{ color: '#8b919d' }}>Movie not found</p>
          <button
            onClick={() => navigate('/')}
            style={{
              marginTop: 16, padding: '8px 20px',
              background: 'rgba(164,201,255,0.12)', color: '#a4c9ff',
              borderRadius: 9999, border: '1px solid rgba(164,201,255,0.3)',
              cursor: 'pointer', fontSize: 13, fontFamily: 'inherit',
            }}
          >
            Back to catalog
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page" style={{ paddingBottom: 80 }}>
      {/* Hero poster with gradient */}
      <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', overflow: 'hidden' }}>
        <img
          src={movie.poster}
          alt={movie.title}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            objectPosition: 'top center',
          }}
          onError={e => { e.target.style.background = '#1c1b1b'; e.target.style.opacity = 0; }}
        />
        <div style={{
          position: 'absolute', inset: 0,
          background: 'linear-gradient(to bottom, rgba(19,19,19,0.2) 0%, rgba(19,19,19,0.7) 70%, rgba(19,19,19,1) 100%)',
        }} />

        {/* Back button */}
        <button
          onClick={() => navigate(-1)}
          style={{
            position: 'absolute',
            top: 16, left: 16,
            width: 36, height: 36,
            borderRadius: '50%',
            background: 'rgba(19,19,19,0.7)',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(72,72,71,0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18, color: '#e5e2e1' }}>
            arrow_back
          </span>
        </button>

        {/* Title block over gradient */}
        <div style={{ position: 'absolute', bottom: 16, left: 16, right: 16 }}>
          <h2 style={{
            fontFamily: "'Manrope', sans-serif",
            fontSize: 22,
            fontWeight: 800,
            color: '#e5e2e1',
            letterSpacing: '-0.3px',
            marginBottom: 8,
            lineHeight: 1.2,
          }}>
            {movie.title}
          </h2>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {movie.rating && movie.rating !== 'N/A' && <Chip label={movie.rating} />}
            <Chip label={String(movie.year)} />
            {movie.runtime && movie.runtime !== 'N/A' && <Chip label={movie.runtime} />}
            {(movie.genre || []).slice(0, 2).map(g => <Chip key={g} label={g} />)}
          </div>
        </div>
      </div>

      {/* Score strip */}
      <div style={{
        display: 'flex',
        padding: '14px 16px',
        gap: 10,
        background: '#1c1b1b',
        borderBottom: '1px solid #262625',
      }}>
        <ScoreStrip label="Chai" score={movie.chaiScore} color="#a4c9ff" seen={movie.chaiSeen} />
        <div style={{ width: 1, background: '#262625', flexShrink: 0 }} />
        <ScoreStrip label="Noel" score={movie.noelScore} color="#ffb4aa" seen={movie.noelSeen} />
      </div>

      {/* Chrome Tab bar */}
      <div style={{ background: '#1c1b1b', position: 'sticky', top: 0, zIndex: 10 }}>
        <ChromeTab tabs={TABS} active={activeTab} onChange={setActiveTab} />
      </div>

      {/* Tab content */}
      <div>
        {activeTab === 'overview' && <OverviewTab movie={movie} />}
        {activeTab === 'chai' && <ChaiTab movie={movie} />}
        {activeTab === 'noel' && <NoelTab movie={movie} />}
      </div>
    </div>
  );
}

function Chip({ label }) {
  return (
    <span style={{
      fontSize: 11,
      color: '#b1c8ed',
      background: 'rgba(177,200,237,0.12)',
      borderRadius: 9999,
      padding: '3px 8px',
      fontWeight: 500,
    }}>
      {label}
    </span>
  );
}

function ScoreStrip({ label, score, color, seen }) {
  const confidence =
    score >= 90 ? 'Top Pick' :
    score >= 80 ? 'Strong Contender' :
    score >= 70 ? 'Likely Enjoy' : 'Mixed Signals';

  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{
        fontSize: 28,
        fontWeight: 700,
        color,
        lineHeight: 1,
        fontFamily: "'Manrope', sans-serif",
      }}>
        {score != null ? `${score}%` : '—'}
      </span>
      <div>
        <p style={{ fontSize: 11, color: '#8b919d', marginBottom: 1 }}>{label}</p>
        <p style={{ fontSize: 12, color, fontWeight: 600 }}>{score != null ? confidence : '—'}</p>
        {seen && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 3, marginTop: 2 }}>
            <span className="material-symbols-outlined" style={{
              fontSize: 11, color: '#8b919d',
              fontVariationSettings: `'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 20`,
            }}>check_circle</span>
            <span style={{ fontSize: 10, color: '#8b919d' }}>Seen</span>
          </div>
        )}
      </div>
    </div>
  );
}
