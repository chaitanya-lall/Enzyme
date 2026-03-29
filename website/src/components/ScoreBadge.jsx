export default function ScoreBadge({ score, size = 36, label }) {
  const color = score >= 85 ? '#a4c9ff' : score >= 70 ? '#b1c8ed' : '#8b919d';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
      <div style={{
        width: size,
        height: size,
        borderRadius: '50%',
        border: `2px solid ${color}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(19,19,19,0.9)',
      }}>
        <span style={{
          fontSize: size * 0.3,
          fontWeight: 700,
          color,
          lineHeight: 1,
        }}>
          {score}
        </span>
      </div>
      {label && (
        <span style={{ fontSize: 9, color: '#8b919d', fontWeight: 600, letterSpacing: '0.04em' }}>
          {label.toUpperCase()}
        </span>
      )}
    </div>
  );
}
