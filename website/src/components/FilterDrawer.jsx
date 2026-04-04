import { useState } from 'react';

const GENRES = [
  'All', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
  'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
  'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
];

const SERVICES = ['All', 'Netflix', 'Prime Video', 'Max', 'Disney+', 'Hulu', 'Apple TV+', 'Peacock', 'Paramount+'];
const SERVICE_KEYS = { 'Netflix': 'netflix', 'Prime Video': 'prime', 'Max': 'max', 'Disney+': 'disney', 'Hulu': 'hulu', 'Apple TV+': 'apple', 'Peacock': 'peacock', 'Paramount+': 'paramount' };

export default function FilterDrawer({ isOpen, onClose, filters, onChange, yearMin, yearMax }) {
  const YEAR_MIN = yearMin ?? 1950;
  const YEAR_MAX = yearMax ?? 2026;

  const [localFilters, setLocalFilters] = useState(filters);

  const toggle = (key, value) => {
    setLocalFilters(f => ({ ...f, [key]: f[key] === value ? 'All' : value }));
  };

  const apply = () => {
    onChange(localFilters);
    onClose();
  };

  const reset = () => {
    const defaultFilters = {
      genre: 'All', service: 'All', type: 'All', sort: 'chai',
      chaiStatus: 'All', noelStatus: 'All',
      imdbMin: 0, yearMin: YEAR_MIN, yearMax: YEAR_MAX,
    };
    setLocalFilters(defaultFilters);
    onChange(defaultFilters);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <>
      <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 200 }} />

      <div style={{
        position: 'fixed', bottom: 0, left: '50%', transform: 'translateX(-50%)',
        width: '100%', maxWidth: 430, background: '#1c1b1b',
        borderTopLeftRadius: 20, borderTopRightRadius: 20,
        zIndex: 201, padding: '20px 20px 100px', maxHeight: '85vh', overflowY: 'auto',
      }}>
        {/* Handle */}
        <div style={{ width: 36, height: 4, background: '#484847', borderRadius: 9999, margin: '0 auto 20px' }} />

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: '#e5e2e1' }}>Filter & Sort</span>
          <button onClick={reset} style={{ fontSize: 13, color: '#a4c9ff', cursor: 'pointer', border: 'none', background: 'none', fontFamily: 'inherit' }}>
            Reset
          </button>
        </div>

        {/* Sort */}
        <Section label="Sort By">
          {[
            { key: 'chai', label: 'Chai Score' },
            { key: 'noel', label: 'Noel Score' },
            { key: 'imdb', label: 'IMDb Score' },
            { key: 'year', label: 'Newest First' },
          ].map(opt => (
            <Chip key={opt.key} label={opt.label} active={localFilters.sort === opt.key}
              onClick={() => setLocalFilters(f => ({ ...f, sort: opt.key }))} />
          ))}
        </Section>

        {/* Chai Watch Status */}
        <Section label="Chai Watch Status">
          {['All', 'Seen', 'Not Seen'].map(s => (
            <Chip key={s} label={s} color="chai"
              active={localFilters.chaiStatus === s}
              onClick={() => setLocalFilters(f => ({ ...f, chaiStatus: s }))} />
          ))}
        </Section>

        {/* Noel Watch Status */}
        <Section label="Noel Watch Status">
          {['All', 'Seen', 'Not Seen'].map(s => (
            <Chip key={s} label={s} color="noel"
              active={localFilters.noelStatus === s}
              onClick={() => setLocalFilters(f => ({ ...f, noelStatus: s }))} />
          ))}
        </Section>

        {/* IMDb Score */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: '#8b919d', letterSpacing: '0.08em' }}>
              IMDB SCORE
            </p>
            <span style={{ fontSize: 12, color: localFilters.imdbMin > 0 ? '#f5c518' : '#484847', fontWeight: 600 }}>
              {localFilters.imdbMin > 0 ? `≥ ${localFilters.imdbMin.toFixed(1)}` : 'Any'}
            </span>
          </div>
          <RangeSlider
            min={0} max={9} step={0.5}
            value={localFilters.imdbMin}
            color="#f5c518"
            onChange={v => setLocalFilters(f => ({ ...f, imdbMin: v }))}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
            <span style={{ fontSize: 10, color: '#484847' }}>Any</span>
            <span style={{ fontSize: 10, color: '#484847' }}>9.0+</span>
          </div>
        </div>

        {/* Release Year */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: '#8b919d', letterSpacing: '0.08em' }}>
              RELEASE YEAR
            </p>
            <span style={{ fontSize: 12, color: '#a4c9ff', fontWeight: 600 }}>
              {localFilters.yearMin === YEAR_MIN && localFilters.yearMax === YEAR_MAX
                ? 'Any'
                : `${localFilters.yearMin} – ${localFilters.yearMax}`}
            </span>
          </div>
          <DualRangeSlider
            min={YEAR_MIN} max={YEAR_MAX}
            low={localFilters.yearMin} high={localFilters.yearMax}
            color="#a4c9ff"
            onChangeLow={v => setLocalFilters(f => ({ ...f, yearMin: Math.min(v, f.yearMax) }))}
            onChangeHigh={v => setLocalFilters(f => ({ ...f, yearMax: Math.max(v, f.yearMin) }))}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
            <span style={{ fontSize: 10, color: '#484847' }}>{YEAR_MIN}</span>
            <span style={{ fontSize: 10, color: '#484847' }}>{YEAR_MAX}</span>
          </div>
        </div>

        {/* Type */}
        <Section label="Type">
          {['All', 'Movies', 'TV Shows'].map(t => (
            <Chip key={t} label={t} active={localFilters.type === t}
              onClick={() => setLocalFilters(f => ({ ...f, type: f.type === t ? 'All' : t }))} />
          ))}
        </Section>

        {/* Genre */}
        <Section label="Genre">
          {GENRES.map(g => (
            <Chip key={g} label={g} active={localFilters.genre === g} onClick={() => toggle('genre', g)} />
          ))}
        </Section>

        {/* Service */}
        <Section label="Streaming Service">
          {SERVICES.map(s => {
            const key = s === 'All' ? 'All' : SERVICE_KEYS[s];
            return (
              <Chip key={s} label={s}
                active={localFilters.service === key}
                onClick={() => setLocalFilters(f => ({ ...f, service: f.service === key ? 'All' : key }))} />
            );
          })}
        </Section>

        {/* Apply */}
        <button onClick={apply} style={{
          width: '100%', padding: '14px', background: '#4d93e5', color: '#fff',
          fontWeight: 700, fontSize: 14, borderRadius: 9999, border: 'none',
          cursor: 'pointer', marginTop: 8, fontFamily: 'inherit',
        }}>
          Apply Filters
        </button>
      </div>
    </>
  );
}

function Section({ label, children }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 11, fontWeight: 600, color: '#8b919d', letterSpacing: '0.08em', marginBottom: 10 }}>
        {label.toUpperCase()}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {children}
      </div>
    </div>
  );
}

function Chip({ label, active, onClick, color = 'chai' }) {
  const accent = color === 'noel' ? '#ffb4aa' : '#a4c9ff';
  const bgAlpha = color === 'noel' ? 'rgba(255,180,170,0.15)' : 'rgba(164,201,255,0.15)';
  return (
    <button onClick={onClick} style={{
      padding: '6px 14px', borderRadius: 9999, fontSize: 12, fontWeight: 500,
      fontFamily: 'inherit', cursor: 'pointer',
      border: `1px solid ${active ? accent : '#484847'}`,
      background: active ? bgAlpha : 'transparent',
      color: active ? accent : '#8b919d',
      transition: 'all 0.15s',
    }}>
      {label}
    </button>
  );
}

function RangeSlider({ min, max, step, value, color, onChange }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ flex: 1, position: 'relative' }}>
      <style>{`
        .enzyme-range { -webkit-appearance: none; appearance: none; width: 100%; height: 4px; border-radius: 9999px; outline: none; cursor: pointer; background: linear-gradient(to right, ${color} 0%, ${color} ${pct}%, #353534 ${pct}%, #353534 100%); }
        .enzyme-range::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: ${color}; cursor: pointer; border: 2px solid #1c1b1b; }
        .enzyme-range::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: ${color}; cursor: pointer; border: 2px solid #1c1b1b; }
      `}</style>
      <input
        type="range" min={min} max={max} step={step} value={value}
        className="enzyme-range"
        onChange={e => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function DualRangeSlider({ min, max, low, high, color, onChangeLow, onChangeHigh }) {
  const range = max - min;
  const lowPct  = ((low  - min) / range) * 100;
  const highPct = ((high - min) / range) * 100;

  return (
    <div style={{ position: 'relative', height: 20, margin: '0 8px' }}>
      <style>{`
        .dual-thumb { -webkit-appearance: none; appearance: none; position: absolute; width: 100%; height: 4px; background: transparent; outline: none; pointer-events: none; top: 50%; transform: translateY(-50%); }
        .dual-thumb::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: ${color}; cursor: pointer; pointer-events: all; border: 2px solid #1c1b1b; box-shadow: 0 0 0 1px ${color}40; }
        .dual-thumb::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: ${color}; cursor: pointer; pointer-events: all; border: 2px solid #1c1b1b; }
      `}</style>
      <div style={{
        position: 'absolute', top: '50%', transform: 'translateY(-50%)',
        left: 0, right: 0, height: 4, borderRadius: 9999, background: '#353534',
      }} />
      <div style={{
        position: 'absolute', top: '50%', transform: 'translateY(-50%)',
        left: `${lowPct}%`, width: `${highPct - lowPct}%`,
        height: 4, borderRadius: 9999, background: color,
      }} />
      <input
        type="range" min={min} max={max} step={1} value={low}
        className="dual-thumb"
        style={{ zIndex: low > max - 1 ? 5 : 3 }}
        onChange={e => onChangeLow(Math.min(Number(e.target.value), high - 1))}
      />
      <input
        type="range" min={min} max={max} step={1} value={high}
        className="dual-thumb"
        style={{ zIndex: 4 }}
        onChange={e => onChangeHigh(Math.max(Number(e.target.value), low + 1))}
      />
    </div>
  );
}
