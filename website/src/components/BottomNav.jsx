import { useLocation, useNavigate } from 'react-router-dom';

export default function BottomNav() {
  const location = useLocation();
  const navigate = useNavigate();

  if (location.pathname.startsWith('/detail/')) return null;

  const active = location.pathname === '/search' ? 'search' : 'recommend';

  return (
    <div style={{
      position: 'fixed',
      bottom: 0, left: 0, right: 0,
      height: 64,
      background: 'rgba(28,27,27,0.92)',
      backdropFilter: 'blur(12px)',
      borderTop: '1px solid #262625',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-around',
      paddingBottom: 'env(safe-area-inset-bottom)',
      zIndex: 100,
    }}>
      <NavButton
        icon="auto_awesome"
        label="Recommend"
        active={active === 'recommend'}
        onClick={() => navigate('/')}
      />
      <NavButton
        icon="search"
        label="Search"
        active={active === 'search'}
        onClick={() => navigate('/search')}
      />
    </div>
  );
}

function NavButton({ icon, label, active, onClick }) {
  const color = active ? '#a4c9ff' : '#8b919d';
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
        background: 'none', border: 'none', cursor: 'pointer',
        padding: '6px 24px',
      }}
    >
      <span className="material-symbols-outlined" style={{
        fontSize: 22, color,
        fontVariationSettings: active ? `'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24` : `'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24`,
      }}>
        {icon}
      </span>
      <span style={{ fontSize: 10, color, fontWeight: active ? 600 : 400, letterSpacing: '0.04em' }}>
        {label}
      </span>
    </button>
  );
}
