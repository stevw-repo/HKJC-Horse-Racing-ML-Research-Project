import { PageHeader } from "@/components/PageHeader";
import { QueryState } from "@/components/QueryState";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useRaceday } from "@/hooks/queries";
import { money, pct } from "@/lib/format";

export function RaceDay() {
  const raceday = useRaceday();
  return (
    <div>
      <PageHeader
        title="Race Day"
        subtitle="Value-staking recommendations for an upcoming card (M7 wires the live odds)."
      />
      <QueryState query={raceday}>
        {(card) => (
          <div className="space-y-5">
            {card.mock ? (
              <div className="flex items-center gap-3 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
                <Badge variant="secondary">MOCK</Badge>
                <span>{card.note}</span>
              </div>
            ) : null}

            {card.races.map((race) => (
              <Card key={race.race_no}>
                <CardHeader>
                  <CardTitle>
                    {card.venue} Race {race.race_no}
                  </CardTitle>
                  <CardDescription>
                    {card.race_date} - {race.status ?? ""} - model {card.model_name}.
                    Recommendations only; the platform never places a bet.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>No.</TableHead>
                        <TableHead>Horse</TableHead>
                        <TableHead className="text-right">WIN %</TableHead>
                        <TableHead className="text-right">PLACE %</TableHead>
                        <TableHead className="text-right">Odds</TableHead>
                        <TableHead className="text-right">EV</TableHead>
                        <TableHead className="text-right">Stake</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {race.runners.map((r) => (
                        <TableRow key={`${race.race_no}-${r.saddle}`}>
                          <TableCell>{r.saddle}</TableCell>
                          <TableCell className="font-medium">{r.name ?? r.horse_id}</TableCell>
                          <TableCell className="text-right">{pct(r.win_prob)}</TableCell>
                          <TableCell className="text-right">{pct(r.place_prob)}</TableCell>
                          <TableCell className="text-right">
                            {r.win_odds != null ? r.win_odds.toFixed(1) : "-"}
                          </TableCell>
                          <TableCell className="text-right">
                            <span
                              className={
                                r.ev != null && r.ev >= 0.05
                                  ? "text-emerald-700"
                                  : "text-muted-foreground"
                              }
                            >
                              {pct(r.ev)}
                            </span>
                          </TableCell>
                          <TableCell className="text-right">
                            {r.stake > 0 ? (
                              <Badge variant="success">{money(r.stake)}</Badge>
                            ) : (
                              <span className="text-muted-foreground">-</span>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </QueryState>
    </div>
  );
}
