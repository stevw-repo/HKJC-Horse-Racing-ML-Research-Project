import { Navigate, Route, Routes } from "react-router-dom";

import { Layout } from "@/components/Layout";
import { BacktestExplorer } from "@/pages/BacktestExplorer";
import { DataHealth } from "@/pages/DataHealth";
import { ExperimentCompare } from "@/pages/ExperimentCompare";
import { RaceDay } from "@/pages/RaceDay";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/health" replace />} />
        <Route path="/health" element={<DataHealth />} />
        <Route path="/backtest" element={<BacktestExplorer />} />
        <Route path="/experiments" element={<ExperimentCompare />} />
        <Route path="/race-day" element={<RaceDay />} />
        <Route path="*" element={<Navigate to="/health" replace />} />
      </Routes>
    </Layout>
  );
}
