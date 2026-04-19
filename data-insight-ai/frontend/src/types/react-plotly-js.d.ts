declare module 'react-plotly.js' {
  import * as React from 'react';

  export interface PlotlyFigureProps {
    data: unknown[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
  }

  const Plot: React.ComponentType<PlotlyFigureProps>;
  export default Plot;
}
