import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { PageHeader } from "@/components/PageHeader";
import { QueryState } from "@/components/QueryState";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useLeaderboard } from "@/hooks/queries";
import { COLORS } from "@/lib/colors";
import { pct } from "@/lib/format";

export function ExperimentCompare() {
  const leaderboard = useLeaderboard();
  return (
    <div>
      <PageHeader
        title="Experiment Compare"
        subtitle="The model zoo ranked on the same honest walk-forward (M3)."
      />
      <QueryState query={leaderboard}>
        {(rows) => {
          if (rows.length === 0) {
            return (
              <Card>
                <CardContent className="p-6">
                  <p className="text-sm text-muted-foreground">
                    No leaderboard yet. Run <code>hkjc train</code> to populate this.
                  </p>
                </CardContent>
              </Card>
            );
          }
          const chart = rows.map((r) => ({
            name: r.name,
            roiPct: r.model_win_roi == null ? 0 : r.model_win_roi * 100,
          }));
          return (
            <div className="space-y-5">
              <Card>
                <CardHeader>
                  <CardTitle>Model-only WIN ROI</CardTitle>
                  <CardDescription>
                    Ranked best to worst. The dashed line is the ~17.5% takeout - no model clears it.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chart} margin={{ top: 8, right: 8, left: -12, bottom: 40 }}>
                      <CartesianGrid stroke={COLORS.grid} vertical={false} />
                      <XAxis
                        dataKey="name"
                        tick={{ fontSize: 10, fill: COLORS.axis }}
                        interval={0}
                        angle={-40}
                        textAnchor="end"
                        height={64}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: COLORS.axis }}
                        tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                      />
                      <Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} />
                      <ReferenceLine y={-17.5} stroke={COLORS.red} strokeDasharray="4 4" />
                      <Bar dataKey="roiPct" fill={COLORS.blue} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Leaderboard</CardTitle>
                  <CardDescription>
                    Models on the grouped within-race likelihood (logit, NNs) calibrate best
                    (low ECE); the pointwise GBMs need the calibration layer.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Model</TableHead>
                        <TableHead className="text-right">Log-loss</TableHead>
                        <TableHead className="text-right">Brier</TableHead>
                        <TableHead className="text-right">Top-1</TableHead>
                        <TableHead className="text-right">ECE</TableHead>
                        <TableHead className="text-right">Model WIN ROI</TableHead>
                        <TableHead className="text-right">Blend WIN ROI</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {rows.map((r) => (
                        <TableRow key={r.name}>
                          <TableCell className="font-medium">{r.name}</TableCell>
                          <TableCell className="text-right">{r.win_log_loss.toFixed(4)}</TableCell>
                          <TableCell className="text-right">{r.brier.toFixed(4)}</TableCell>
                          <TableCell className="text-right">{pct(r.top1_hit_rate)}</TableCell>
                          <TableCell className="text-right">{r.ece.toFixed(4)}</TableCell>
                          <TableCell className="text-right">{pct(r.model_win_roi)}</TableCell>
                          <TableCell className="text-right">{pct(r.blend_win_roi)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          );
        }}
      </QueryState>
    </div>
  );
}
