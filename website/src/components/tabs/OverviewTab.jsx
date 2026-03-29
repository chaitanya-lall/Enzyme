export default function OverviewTab({ movie }) {
  const parental = movie.parentalTags || { sex: 0, violence: 0, profanity: 0, drugs: 0, frightening: 0 };
  const maxLevel = 4;
  const cast = movie.cast || [];

  return (
    <div style={{ padding: '20px 16px' }}>

      {/* Synopsis */}
      {movie.synopsis && (
        <Section label="The Premise">
          <p style={{ fontSize: 14, color: '#b1c8ed', lineHeight: 1.65 }}>
            {movie.synopsis}
          </p>
        </Section>
      )}

      {/* Director + Cast */}
      <Section label="Filmmakers">
        {movie.director && movie.director !== 'N/A' && (
          <MetaRow icon="movie_creation" label="Director" value={movie.director} />
        )}
        {cast.length > 0 && (
          <MetaRow icon="groups" label="Cast" value={cast.join(', ')} />
        )}
      </Section>

      {/* Ratings */}
      <Section label="Critical Reception">
        <div style={{ display: 'flex', gap: 12 }}>
          <RatingBadge source="IMDb" score={movie.imdbScore != null ? String(movie.imdbScore) : '—'} color="#f5c518" />
          <RatingBadge source="RT"   score={movie.rtScore   != null ? `${movie.rtScore}%`   : '—'} color="#fa320a" />
          <RatingBadge source="Meta" score={movie.metaScore != null ? String(movie.metaScore): '—'} color="#6c3" />
        </div>
      </Section>

      {/* Awards */}
      {movie.awards && (
        <Section label="Recognition">
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 14px',
            background: 'rgba(255, 180, 170, 0.1)',
            border: '1px solid rgba(255,180,170,0.3)',
            borderRadius: 9999,
          }}>
            <span className="material-symbols-outlined" style={{ fontSize: 16, color: '#ffb4aa' }}>
              emoji_events
            </span>
            <span style={{ fontSize: 13, color: '#ffb4aa', fontWeight: 600 }}>
              {movie.awards}
            </span>
          </div>
        </Section>
      )}

      {/* Parents Guide */}
      <Section label="Parents Guide">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <ParentalRow label="Sex & Nudity"    level={parental.sex}        max={maxLevel} />
          <ParentalRow label="Violence & Gore" level={parental.violence}   max={maxLevel} />
          <ParentalRow label="Profanity"       level={parental.profanity}  max={maxLevel} />
          <ParentalRow label="Alcohol & Drugs" level={parental.drugs}      max={maxLevel} />
          <ParentalRow label="Frightening"     level={parental.frightening}max={maxLevel} />
        </div>
      </Section>
    </div>
  );
}

function Section({ label, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <p style={{
        fontSize: 11, fontWeight: 600, color: '#8b919d',
        letterSpacing: '0.08em', marginBottom: 12,
      }}>
        {label.toUpperCase()}
      </p>
      {children}
    </div>
  );
}

function MetaRow({ icon, label, value }) {
  return (
    <div style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'flex-start' }}>
      <span className="material-symbols-outlined" style={{ fontSize: 16, color: '#8b919d', flexShrink: 0, marginTop: 2 }}>
        {icon}
      </span>
      <div>
        <span style={{ fontSize: 11, color: '#8b919d', display: 'block', marginBottom: 1 }}>{label}</span>
        <span style={{ fontSize: 13, color: '#e5e2e1' }}>{value}</span>
      </div>
    </div>
  );
}

function RatingBadge({ source, score, color }) {
  return (
    <div style={{
      flex: 1, background: '#1c1b1b', borderRadius: 10,
      padding: '12px 10px', textAlign: 'center',
    }}>
      <p style={{ fontSize: 18, fontWeight: 700, color, marginBottom: 2 }}>{score}</p>
      <p style={{ fontSize: 10, color: '#8b919d', fontWeight: 600, letterSpacing: '0.06em' }}>{source}</p>
    </div>
  );
}

function ParentalRow({ label, level, max }) {
  const labelMap = ['None', 'Mild', 'Moderate', 'Strong', 'Severe'];
  const colorMap = ['#8b919d', '#a4c9ff', '#ffb4aa', '#ff6464', '#ff3333'];
  const safeLevel = level ?? 0;
  const color = colorMap[safeLevel] ?? '#8b919d';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{ fontSize: 12, color: '#8b919d', width: 120, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, display: 'flex', gap: 3 }}>
        {Array.from({ length: max }).map((_, i) => (
          <div key={i} style={{
            flex: 1, height: 4, borderRadius: 9999,
            background: i < safeLevel ? color : '#262625',
          }} />
        ))}
      </div>
      <span style={{ fontSize: 11, color, width: 54, textAlign: 'right', flexShrink: 0 }}>
        {labelMap[safeLevel] ?? 'None'}
      </span>
    </div>
  );
}
