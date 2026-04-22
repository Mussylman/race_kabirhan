import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { PublicDisplay } from './pages/PublicDisplay';
import { OperatorPanel } from './pages/OperatorPanel';
import { AnalyticsLayout } from './pages/analytics/AnalyticsLayout';
import { TimelineView } from './pages/analytics/TimelineView';
import { ReIDLabeler } from './pages/analytics/ReIDLabeler';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PublicDisplay />} />
        <Route path="/operator" element={<OperatorPanel />} />
        <Route path="/analytics" element={<AnalyticsLayout />}>
          <Route index element={<AnalyticsLanding />} />
          <Route path=":recId">
            <Route index element={<Navigate to="timeline" replace />} />
            <Route path="timeline" element={<TimelineView />} />
            <Route path="labeler" element={<ReIDLabeler />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

const AnalyticsLanding = () => (
  <div className="p-8 text-white/70">
    <h1 className="text-2xl font-semibold mb-4">Analytics</h1>
    <p>Pick a recording from the sidebar to open Timeline or ReID Labeler.</p>
    <p className="mt-2 text-sm text-white/50">
      If the sidebar is empty, index a recording:
    </p>
    <pre className="mt-2 bg-black/40 p-3 rounded text-xs overflow-auto">
{`curl -X POST http://localhost:8000/analytics/recordings/index \\
     -H 'Content-Type: application/json' \\
     -d '{"rec_id":"rec2_yaris","jsonl_dir":"/tmp/logs2/frames","mp4_dir":"/path/to/recording2"}'`}
    </pre>
  </div>
);

export default App;
