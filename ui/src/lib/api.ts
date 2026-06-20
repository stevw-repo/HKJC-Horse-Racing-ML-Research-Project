import type {
  Backtest,
  Health,
  LeaderboardRow,
  RaceDay,
  RaceSummary,
  StakingRow,
} from "@/types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => get<Health>("/api/health"),
  backtest: () => get<Backtest>("/api/backtest"),
  leaderboard: () => get<LeaderboardRow[]>("/api/leaderboard"),
  staking: () => get<StakingRow[]>("/api/staking"),
  races: (limit = 50) => get<RaceSummary[]>(`/api/races?limit=${limit}`),
  raceday: () => get<RaceDay>("/api/raceday"),
};
