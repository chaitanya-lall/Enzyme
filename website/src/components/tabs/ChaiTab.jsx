export default function ChaiTab({ movie }) {
  const chai = movie.chai || {};
  const score = movie.chaiScore;
  const color = score >= 85 ? '#a4c9ff' : score >= 70 ? '#b1c8ed' : '#8b919d';

  const r = 42, cx = 56, cy = 56;
  const circumference = 2 * Math.PI * r;
  const filled = ((score ?? 0) / 100) * circumference;

  const confidence = chai.confidence || (
    score >= 90 ? 'Top Pick' : score >= 80 ? 'Strong Contender' :
    score >= 70 ? 'Likely Enjoy' : 'Mixed Signals'
  );

  const confidenceColor = {
    'Top Pick': '#a4c9ff', 'Must Watch': '#a4c9ff',
    'Strong Contender': '#b1c8ed', 'Solid Pick': '#b1c8ed',
    'Likely Enjoy': '#8b919d',
  }[confidence] ?? '#8b919d';

  return (
    <div style={{ padding: '20px 16px' }}>

      {/* Score ring */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 20,
        background: '#1c1b1b', borderRadius: 16, padding: '20px', marginBottom: 20,
      }}>
        <svg width={112} height={112} viewBox="0 0 112 112">
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="#262625" strokeWidth={8} />
          <circle
            cx={cx} cy={cy} r={r}
            fill="none" stroke={color} strokeWidth={8} strokeLinecap="round"
            strokeDasharray={`${filled} ${circumference}`}
            strokeDashoffset={circumference * 0.25}
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ transition: 'stroke-dasharray 0.6s ease' }}
          />
          <text x={cx} y={cy - 6} textAnchor="middle" fill={color} fontSize={22} fontWeight={700} fontFamily="Inter">
            {score != null ? `${score}%` : '—'}
          </text>
          <text x={cx} y={cy + 12} textAnchor="middle" fill="#8b919d" fontSize={10} fontFamily="Inter">
            MATCH
          </text>
        </svg>

        <div style={{ flex: 1 }}>
          <p style={{ fontSize: 11, color: '#8b919d', letterSpacing: '0.06em', marginBottom: 4 }}>
            CHAI'S VERDICT
          </p>
          <p style={{ fontSize: 16, fontWeight: 700, color: confidenceColor, marginBottom: 8 }}>
            {confidence}
          </p>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 10px',
            background: movie.chaiSeen ? 'rgba(164,201,255,0.1)' : 'rgba(139,145,157,0.1)',
            borderRadius: 9999,
          }}>
            <span className="material-symbols-outlined" style={{
              fontSize: 12, color: movie.chaiSeen ? '#a4c9ff' : '#8b919d',
            }}>
              {movie.chaiSeen ? 'check_circle' : 'radio_button_unchecked'}
            </span>
            <span style={{ fontSize: 11, color: movie.chaiSeen ? '#a4c9ff' : '#8b919d', fontWeight: 500 }}>
              {movie.chaiSeen ? 'Chai has seen this' : 'Not yet watched'}
            </span>
          </div>
        </div>
      </div>

      {/* Narrative — only shown when available */}
      {chai.narrative && (
        <Section label="Why This Score">
          <div style={{
            background: '#1c1b1b', borderRadius: 12,
            padding: '14px 16px', borderLeft: `3px solid ${color}`,
          }}>
            <p style={{ fontSize: 13, color: '#b1c8ed', lineHeight: 1.7 }}>
              {chai.narrative}
            </p>
          </div>
        </Section>
      )}

      {/* Key Drivers — only shown when available */}
      {chai.drivers && chai.drivers.length > 0 && (
        <Section label="Key Drivers">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {chai.drivers.map((d, i) => (
              <DriverChip key={i} label={d.label} impact={d.impact} />
            ))}
          </div>
        </Section>
      )}

      {/* Closest Match */}
      {chai.closestMatch && (
        <Section label="Closest Match in History">
          <div style={{
            background: '#1c1b1b', borderRadius: 12, padding: '14px 16px',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{
              width: 40, height: 40, borderRadius: '50%',
              background: 'rgba(164,201,255,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
            }}>
              <span className="material-symbols-outlined" style={{ fontSize: 20, color: '#a4c9ff' }}>link</span>
            </div>
            <div style={{ flex: 1 }}>
              <p style={{ fontSize: 14, fontWeight: 600, color: '#e5e2e1', marginBottom: 2 }}>
                {chai.closestMatch.title}
              </p>
              <p style={{ fontSize: 12, color: '#8b919d' }}>
                Chai rated <span style={{ color: '#a4c9ff' }}>{chai.closestMatch.score}/10</span>
                {' '}· <span style={{ color: '#a4c9ff' }}>{chai.closestMatch.matchPct}%</span> similar profile
              </p>
            </div>
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({ label, children }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 11, fontWeight: 600, color: '#8b919d', letterSpacing: '0.08em', marginBottom: 10 }}>
        {label.toUpperCase()}
      </p>
      {children}
    </div>
  );
}

function DriverChip({ label, impact }) {
  const pos = impact === 'pos';
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 5, padding: '5px 10px',
      borderRadius: 9999,
      background: pos ? 'rgba(164,201,255,0.1)' : 'rgba(255,100,100,0.08)',
      border: `1px solid ${pos ? 'rgba(164,201,255,0.25)' : 'rgba(255,100,100,0.2)'}`,
    }}>
      <span className="material-symbols-outlined" style={{
        fontSize: 12, color: pos ? '#a4c9ff' : '#ff6464',
        fontVariationSettings: `'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 20`,
      }}>
        {pos ? 'arrow_upward' : 'arrow_downward'}
      </span>
      <span style={{ fontSize: 11, color: pos ? '#a4c9ff' : '#ff9090', fontWeight: 500 }}>
        {label}
      </span>
    </div>
  );
}
