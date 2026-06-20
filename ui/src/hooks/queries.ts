import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: api.health });
export const useBacktest = () => useQuery({ queryKey: ["backtest"], queryFn: api.backtest });
export const useLeaderboard = () => useQuery({ queryKey: ["leaderboard"], queryFn: api.leaderboard });
export const useStaking = () => useQuery({ queryKey: ["staking"], queryFn: api.staking });
export const useRaceday = () => useQuery({ queryKey: ["raceday"], queryFn: api.raceday });
export const useRaces = (limit = 50) =>
  useQuery({ queryKey: ["races", limit], queryFn: () => api.races(limit) });
