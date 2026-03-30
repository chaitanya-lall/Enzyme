import { BrowserRouter, Routes, Route } from 'react-router-dom';
import RecommendPage from './pages/RecommendPage';
import SearchPage from './pages/SearchPage';
import DetailPage from './pages/DetailPage';
import BottomNav from './components/BottomNav';

function App() {
  return (
    <BrowserRouter>
      <div style={{ paddingBottom: 64 }}>
        <Routes>
          <Route path="/" element={<RecommendPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/detail/:id" element={<DetailPage />} />
        </Routes>
      </div>
      <BottomNav />
    </BrowserRouter>
  );
}

export default App;
