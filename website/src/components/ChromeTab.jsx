export default function ChromeTab({ tabs, active, onChange }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-end',
      background: '#1c1b1b',
      padding: '0 8px',
      borderBottom: '1px solid #262625',
      position: 'relative',
    }}>
      {tabs.map((tab) => {
        const isActive = tab.key === active;
        return (
          <button
            key={tab.key}
            onClick={() => onChange(tab.key)}
            style={{
              position: 'relative',
              height: 40,
              padding: '0 20px',
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.05em',
              color: isActive ? '#a4c9ff' : '#8b919d',
              background: isActive ? '#131313' : 'transparent',
              border: 'none',
              cursor: 'pointer',
              borderTopLeftRadius: isActive ? 12 : 0,
              borderTopRightRadius: isActive ? 12 : 0,
              transition: 'color 0.15s',
              fontFamily: "'Inter', sans-serif",
              whiteSpace: 'nowrap',
            }}
          >
            {tab.label.toUpperCase()}
            {isActive && (
              <>
                {/* Left curved connector */}
                <span style={{
                  position: 'absolute',
                  bottom: 0,
                  left: -8,
                  width: 8,
                  height: 8,
                  background: 'transparent',
                  boxShadow: '4px 4px 0 4px #131313',
                  borderBottomRightRadius: 8,
                  pointerEvents: 'none',
                }} />
                {/* Right curved connector */}
                <span style={{
                  position: 'absolute',
                  bottom: 0,
                  right: -8,
                  width: 8,
                  height: 8,
                  background: 'transparent',
                  boxShadow: '-4px 4px 0 4px #131313',
                  borderBottomLeftRadius: 8,
                  pointerEvents: 'none',
                }} />
              </>
            )}
          </button>
        );
      })}
    </div>
  );
}
