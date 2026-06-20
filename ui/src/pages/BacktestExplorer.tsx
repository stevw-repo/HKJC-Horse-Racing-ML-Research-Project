import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { PageHeader } from "@/components/PageHeader";
import { QueryState } from "@/components/QueryState";
import { Stat } from "@/components/Stat";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useBacktest, useStaking } from "@/hooks/queries";
import { COLORS } from "@/lib/colors";
import { money, num, pct } from "@/lib/format";
import type { StakingRow } from "@/types";

const POLICY_LABELS: Record<string, string> = {
  model_win: "Model-only WIN",
  model_place: "Model-only PLACE",
  blend_win: "Market-blended WIN",
};

export function BacktestExplorer() {
  const backtest = useBacktest();
  const staking = useStaking();
  return (
    <div>
      <PageHeader
        title="Backtest Explorer"
        subtitle="Honest walk-forward results (M2) + the multi-bankroll staking sweep (M5)."
      />
      <QueryState query={backtest}>
        {(bt) => (
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <Stat
                label="OOS races"
                value={num(bt.n_oos_races)}
                sub={`${bt.test_span[0]} .. ${bt.test_span[1]}`}
              />
              <Stat label="WIN log-loss" value={bt.win_log_loss.toFixed(4)} sub={`feature ${bt.feature_version}`} />
              <Stat label="Top-1 hit" value={pct(bt.top1_hit_rate)} />
              <Stat label="Calibration (ECE)" value={bt.ece.toFixed(4)} />
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Betting policies</CardTitle>
                <CardDescription>
                  Paid at the stored final dividends. No staking rule beats the ~17.5% takeout
                  (PLAN.md section 1F).
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Policy</TableHead>
                      <TableHead className="text-right">Bets</TableHead>
                      <TableHead className="text-right">ROI</TableHead>
                      <TableHead className="text-right">95% CI</TableHead>
                      <TableHead className="text-right">Sharpe</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(bt.policies).map(([key, p]) => (
                      <TableRow key={key}>
                        <TableCell className="font-medium">{POLICY_LABELS[key] ?? p.name}</TableCell>
                        <TableCell className="text-right">{num(p.n_bets)}</TableCell>
                        <TableCell className="text-right">{pct(p.roi)}</TableCell>
                        <TableCell className="text-right text-muted-foreground">
                          [{pct(p.roi_lo)}, {pct(p.roi_hi)}]
                        </TableCell>
                        <TableCell className="text-right">{p.sharpe.toFixed(3)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {bt.canary_coef_ratio != null ? (
                  <p className="mt-3 text-xs text-muted-foreground">
                    Leakage canary: coef ratio {bt.canary_coef_ratio.toFixed(4)} (must be ~0) ;
                    random-pick sentinel ROI {pct(bt.canary_roi)}.
                  </p>
                ) : null}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>WIN calibration</CardTitle>
                <CardDescription>Predicted vs observed win rate (the diagonal is perfect).</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={bt.calibration}
                    margin={{ top: 8, right: 12, left: -8, bottom: 4 }}
                  >
                    <CartesianGrid stroke={COLORS.grid} />
                    <XAxis
                      type="number"
                      dataKey="pred_mean"
                      domain={[0, 1]}
                      tick={{ fontSize: 11, fill: COLORS.axis }}
                      tickFormatter={(v: number) => v.toFixed(1)}
                    />
                    <YAxis
                      type="number"
                      domain={[0, 1]}
                      tick={{ fontSize: 11, fill: COLORS.axis }}
                      tickFormatter={(v: number) => v.toFixed(1)}
                    />
                    <Tooltip formatter={(value) => Number(value).toFixed(3)} />
                    <ReferenceLine
                      segment={[
                        { x: 0, y: 0 },
                        { x: 1, y: 1 },
                      ]}
                      stroke={COLORS.axis}
                      strokeDasharray="4 4"
                    />
                    <Line
                      type="monotone"
                      dataKey="obs_rate"
                      stroke={COLORS.blue}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <QueryState query={staking}>{(rows) => <StakingSection rows={rows} />}</QueryState>
          </div>
        )}
      </QueryState>
    </div>
  );
}

function StakingSection({ rows }: { rows: StakingRow[] }) {
  const bankrolls = useMemo(
    () => Array.from(new Set(rows.map((r) => r.bankroll))).sort((a, b) => a - b),
    [rows],
  );
  const [bankroll, setBankroll] = useState<number | null>(null);
  if (rows.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Staking sweep</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No sweep yet. Run <code>hkjc risk sweep</code> to populate this.
          </p>
        </CardContent>
      </Card>
    );
  }
  const active = bankroll ?? bankrolls[0];
  const cells = rows
    .filter((r) => r.bankroll === active)
    .map((r) => ({ ...r, roiPct: r.roi * 100 }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Staking sweep (M5)</CardTitle>
        <CardDescription>ROI by policy at the selected bankroll.</CardDescription>
        <div className="flex flex-wrap gap-2 pt-2">
          {bankrolls.map((b) => (
            <Button
              key={b}
              size="sm"
              variant={b === active ? "default" : "outline"}
              onClick={() => setBankroll(b)}
            >
              {money(b)}
            </Button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={cells} margin={{ top: 8, right: 8, left: -12, bottom: 40 }}>
            <CartesianGrid stroke={COLORS.grid} vertical={false} />
            <XAxis
              dataKey="policy"
              tick={{ fontSize: 9, fill: COLORS.axis }}
              interval={0}
              angle={-45}
              textAnchor="end"
              height={56}
            />
            <YAxis
              tick={{ fontSize: 11, fill: COLORS.axis }}
              tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            />
            <Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} />
            <ReferenceLine y={-17.5} stroke={COLORS.red} strokeDasharray="4 4" />
            <Bar dataKey="roiPct" fill={COLORS.primary} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="destructive">-17.5%</Badge>
          dashed line = the takeout. Every policy sits at or below it.
        </div>
      </CardContent>
    </Card>
  );
}
