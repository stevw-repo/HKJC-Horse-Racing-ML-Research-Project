import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";

// Renders loading / error states, then hands the resolved data to `children`.
export function QueryState<T>({
  query,
  children,
}: {
  query: UseQueryResult<T>;
  children: (data: T) => ReactNode;
}) {
  if (query.isLoading) {
    return <div className="p-10 text-center text-sm text-muted-foreground">Loading...</div>;
  }
  if (query.isError) {
    return (
      <div className="p-10 text-center text-sm text-destructive">
        {(query.error as Error).message}
      </div>
    );
  }
  if (query.data === undefined) {
    return <div className="p-10 text-center text-sm text-muted-foreground">No data.</div>;
  }
  return <>{children(query.data)}</>;
}
