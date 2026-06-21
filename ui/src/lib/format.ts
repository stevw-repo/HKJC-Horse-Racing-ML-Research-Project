export function pct(x: number | null | undefined, digits = 1): string {
  return x == null ? "-" : `${(x * 100).toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 0): string {
  return x == null
    ? "-"
    : x.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function money(x: number | null | undefined): string {
  return x == null ? "-" : `HK$${num(Math.round(x))}`;
}
