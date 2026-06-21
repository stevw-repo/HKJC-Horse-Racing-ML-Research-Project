import { Activity, BarChart3, CalendarClock, FlaskConical } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

import { cn } from "@/lib/utils";

const NAV = [
  { to: "/health", label: "Data Health", icon: Activity },
  { to: "/backtest", label: "Backtest Explorer", icon: BarChart3 },
  { to: "/experiments", label: "Experiment Compare", icon: FlaskConical },
  { to: "/race-day", label: "Race Day", icon: CalendarClock },
];

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-muted/30">
      <aside className="flex w-60 shrink-0 flex-col border-r bg-background p-4">
        <div className="px-2 pb-5">
          <h1 className="text-base font-bold">HKJC Research</h1>
          <p className="text-xs text-muted-foreground">Recommends only - never bets</p>
        </div>
        <nav className="space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                )
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <p className="mt-auto px-2 pt-4 text-[11px] leading-snug text-muted-foreground">
          Local research sandbox. No code path submits a wager.
        </p>
      </aside>
      <main className="flex-1 overflow-auto p-6">{children}</main>
    </div>
  );
}
