import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { PageHeader } from "@/components/PageHeader";
import { QueryState } from "@/components/QueryState";
import { Stat } from "@/components/Stat";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useHealth, useRaces } from "@/hooks/queries";
import { COLORS } from "@/lib/colors";
import { num } from "@/lib/format";
import type { Health } from "@/types";

const SOURCES: { key: keyof Health; label: string }[] = [
  { key: "races_rows", label: "Races" },
  { key: "results_rows", label: "Runners" },
  { key: "dividends_rows", label: "Dividends" },
  { key: "horses_rows", label: "Horses" },
  { key: "horse_form_rows", label: "Horse form" },
  { key: "people_rows", label: "Jockeys / trainers" },
  { key: "weather_rows", label: "Weather" },
  { key: "public_holidays_rows", label: "Public holidays" },
  { key: "barrier_trials_rows", label: "Barrier trials" },
  { key: "trackwork_rows", label: "Trackwork" },
  { key: "sectionals_rows", label: "Sectionals" },
  { key: "comments_on_running_rows", label: "Comments (NLP)" },
];

export function DataHealth() {
  const health = useHealth();
  const races = useRaces(12);
  return (
    <div>
      <PageHeader
        title="Data Health"
        subtitle="Stored coverage across every scraped source (M1)."
      />
      <QueryState query={health}>
        {(h) => {
          const seasons = Object.entries(h.seasons)
            .map(([season, meetings]) => ({ season, meetings }))
            .sort((a, b) => a.season.localeCompare(b.season));
          return (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <Stat
                  label="Meetings"
                  value={num(h.meetings)}
                  sub={`${h.date_min ?? "?"} -> ${h.date_max ?? "?"}`}
                />
                <Stat label="Runners" value={num(h.results_rows)} />
                <Stat label="Sectionals" value={num(h.sectionals_rows)} />
                <Stat label="Comments" value={num(h.comments_on_running_rows)} />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Meetings per season</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={seasons} margin={{ top: 4, right: 8, left: -18, bottom: 4 }}>
                      <CartesianGrid stroke={COLORS.grid} vertical={false} />
                      <XAxis
                        dataKey="season"
                        tick={{ fontSize: 10, fill: COLORS.axis }}
                        interval={0}
                        angle={-45}
                        textAnchor="end"
                        height={56}
                      />
                      <YAxis tick={{ fontSize: 11, fill: COLORS.axis }} />
                      <Tooltip />
                      <Bar dataKey="meetings" fill={COLORS.primary} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <div className="grid gap-5 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Source coverage</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Source</TableHead>
                          <TableHead className="text-right">Rows</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {SOURCES.map((s) => (
                          <TableRow key={s.key}>
                            <TableCell>{s.label}</TableCell>
                            <TableCell className="text-right">{num(h[s.key] as number)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Recent races</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <QueryState query={races}>
                      {(rows) => (
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Date</TableHead>
                              <TableHead>Venue</TableHead>
                              <TableHead>Race</TableHead>
                              <TableHead className="text-right">Field</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {rows.map((r) => (
                              <TableRow key={`${r.race_date}-${r.venue}-${r.race_no}`}>
                                <TableCell>{r.race_date}</TableCell>
                                <TableCell>{r.venue}</TableCell>
                                <TableCell>
                                  R{r.race_no}
                                  {r.distance ? ` - ${r.distance}m` : ""}
                                </TableCell>
                                <TableCell className="text-right">{r.field_size}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      )}
                    </QueryState>
                  </CardContent>
                </Card>
              </div>
            </div>
          );
        }}
      </QueryState>
    </div>
  );
}
