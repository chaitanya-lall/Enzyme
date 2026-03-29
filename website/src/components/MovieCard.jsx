import { useNavigate } from 'react-router-dom';
import ScoreBadge from './ScoreBadge';

export default function MovieCard({ movie }) {
  const navigate = useNavigate();

  return (
    <div
      onClick={() => navigate(`/detail/${movie.id}`)}
      style={{
        background: '#1c1b1b',
        borderRadius: 12,
        overflow: 'hidden',
        cursor: 'pointer',
        position: 'relative',
        transition: 'transform 0.15s',
      }}
      onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.02)'}
      onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
    >
      {/* Poster */}
      <div style={{ position: 'relative', aspectRatio: '2/3', overflow: 'hidden' }}>
        <img
          src={movie.poster}
          alt={movie.title}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          onError={e => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
        />
        <div style={{
          display: 'none',
          position: 'absolute', inset: 0,
          background: '#262625',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 8,
        }}>
          <span className="material-symbols-outlined" style={{ color: '#8b919d', fontSize: 32 }}>movie</span>
          <span style={{ color: '#8b919d', fontSize: 11 }}>{movie.title}</span>
        </div>

        {/* Gradient overlay */}
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          height: '50%',
          background: 'linear-gradient(to bottom, transparent, rgba(19,19,19,0.95))',
        }} />

        {/* Scores overlay */}
        <div style={{
          position: 'absolute',
          bottom: 8, left: 8, right: 8,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
        }}>
          <ScoreBadge score={movie.chaiScore} size={32} label="Chai" />
          <ScoreBadge score={movie.noelScore} size={32} label="Noel" />
        </div>
      </div>

      {/* Info */}
      <div style={{ padding: '8px 10px 10px' }}>
        <p style={{
          fontSize: 12,
          fontWeight: 600,
          color: '#e5e2e1',
          lineHeight: 1.3,
          marginBottom: 4,
          overflow: 'hidden',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
        }}>
          {movie.title}
        </p>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: '#8b919d' }}>{movie.year}</span>
          <span style={{ fontSize: 11, color: '#484847' }}>·</span>
          <span style={{
            fontSize: 10,
            color: '#8b919d',
            border: '1px solid #484847',
            borderRadius: 4,
            padding: '1px 4px',
          }}>
            {movie.rating}
          </span>
        </div>
        {/* Genre chips */}
        <div style={{ display: 'flex', gap: 4, marginTop: 6, flexWrap: 'wrap' }}>
          {movie.genre.slice(0, 2).map(g => (
            <span key={g} style={{
              fontSize: 10,
              color: '#b1c8ed',
              background: 'rgba(177,200,237,0.12)',
              borderRadius: 9999,
              padding: '2px 7px',
            }}>
              {g}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
