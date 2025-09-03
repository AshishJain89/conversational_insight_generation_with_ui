import { Card, CardContent } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Legend,
  CartesianGrid,
  ScatterChart,
  Scatter,
} from "recharts";

interface ChartRendererProps {
	columns: string[];
	rows: (string | number | null)[][];
	suggestion?: {
		type: 'line' | 'bar' | 'pie' | 'grouped_bar' | 'scatter' | 'table';
		x?: string | null;
		y?: string | string[] | null;
		series?: string | null;
		reason?: string | null;
	};
}

const COLORS = [
  "#8884d8", "#82ca9d", "#ffc658", "#ff7f50", "#00c49f", "#0088fe",
  "#ffbb28", "#ff4444", "#a0a0ff", "#7ad3ff", "#ff9fb3", "#80e0a7",
];

const METRIC_NAME_HINTS = [
  "count","total","sum","sales","amount","revenue","value","qty","quantity","price","cost","profit"
];

const DATE_NAME_HINTS = [
  "date","time","month","year","quarter","day","created","updated","timestamp"
];

function isNumeric(val: any): boolean {
  if (val === null || val === undefined || val === "") return false;
  const n = Number(val);
  return Number.isFinite(n) && !isNaN(n);
}

function isDateLike(val: any): boolean {
  if (val === null || val === undefined || val === "") return false;
  const s = String(val).trim();
  
  // Check common date patterns
  if (/^\d{4}(-\d{2}){0,2}$/.test(s)) return true;
  if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(s)) return true;
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(s)) return true;
  
  const d = new Date(s);
  return !isNaN(d.getTime()) && d.getFullYear() > 1900;
}

function colNameLooksDate(name: string): boolean {
  const n = name.toLowerCase();
  return DATE_NAME_HINTS.some(hint => n.includes(hint));
}

function colNameLooksMetric(name: string): boolean {
  const n = name.toLowerCase();
  return METRIC_NAME_HINTS.some(hint => n.includes(hint));
}

function getColumnType(column: string, data: Record<string, any>[]): 'numeric' | 'date' | 'categorical' {
  // Check column name first
  if (colNameLooksDate(column)) {
    const vals = data.map(r => r[column]).filter(v => v !== null && v !== undefined);
    if (vals.length > 0 && vals.filter(isDateLike).length / vals.length >= 0.5) {
      return 'date';
    }
  }

  // Check actual values
  const nonNullValues = data.map(r => r[column]).filter(v => v !== null && v !== undefined && v !== "");
  if (nonNullValues.length === 0) return 'categorical';

  const numericCount = nonNullValues.filter(isNumeric).length;
  const numericRatio = numericCount / nonNullValues.length;
  
  if (numericRatio >= 0.8) return 'numeric';
  
  const dateCount = nonNullValues.filter(isDateLike).length;
  const dateRatio = dateCount / nonNullValues.length;
  
  if (dateRatio >= 0.6 || colNameLooksDate(column)) return 'date';
  
  return 'categorical';
}

function getCardinality(column: string, data: Record<string, any>[]): number {
  const uniqueValues = new Set(data.map(r => r[column]));
  return uniqueValues.size;
}

function buildRowObjects(columns: string[], rows: (string|number|null)[][]): Record<string, any>[] {
  return rows.map((row) => {
    const obj: Record<string, any> = {};
    columns.forEach((col, i) => {
      const v = row[i];
      obj[col] = v === null ? "NULL" : v;
    });
    return obj;
  });
}

function topNWithOthers<T extends Record<string, any>>(
  data: T[], sortKey: string, n: number, nameKey: string
): T[] {
  if (!data.length) return data;
  const sorted = [...data].sort((a,b) => Number(b[sortKey]) - Number(a[sortKey]));
  if (sorted.length <= n) return sorted;
  
  const head = sorted.slice(0, n);
  const tail = sorted.slice(n);
  const others = tail.reduce((acc, d) => acc + Number(d[sortKey] || 0), 0);
  return [...head, { [nameKey]: "Others", [sortKey]: others } as T];
}

export const ChartRenderer = ({ columns, rows, suggestion }: ChartRendererProps) => {
  if (!rows || rows.length === 0 || columns.length === 0) return null;

  const data = buildRowObjects(columns, rows);

  // If backend provided a suggestion, render it verbatim
  if (suggestion && suggestion.type) {
    const type = suggestion.type;
    const x = suggestion.x || undefined;
    const y = suggestion.y as string | string[] | undefined;
    const series = suggestion.series || undefined;

    // Guard: ensure referenced columns exist
    const hasCol = (c?: string) => !!c && columns.includes(c);
    const pickYArray = Array.isArray(y) ? y.filter((c) => hasCol(c)).slice(0, 4) : (hasCol(y as string) ? [y as string] : []);

    // Table fallback if mapping is unusable
    if (type !== 'table' && ((type === 'line' || type === 'bar' || type === 'grouped_bar') && pickYArray.length === 0)) {
      return null;
    }

    if (type === 'line' && hasCol(x) && pickYArray.length >= 1) {
      const sorted = [...data].sort((a, b) => new Date(a[x as string]).getTime() - new Date(b[x as string]).getTime());
      return (
        <Card className='mt-4'>
          <CardContent className='h-[300px]'>
            <ResponsiveContainer width='100%' height='100%'>
              <LineChart data={sorted}>
                <CartesianGrid strokeDasharray='3 3' />
                <XAxis dataKey={x as string} />
                <YAxis />
                <Tooltip />
                <Legend />
                {pickYArray.map((yk, i) => (
                  <Line key={yk} type='monotone' dataKey={yk} stroke={COLORS[i % COLORS.length]} dot={false} />
                ))}
              </LineChart> 
            </ResponsiveContainer>
          </CardContent>
        </Card>
      );
    }
    
    if (type === 'scatter' && Array.isArray(y) === false && hasCol(x) && hasCol(y as string)) {
      return (
        <Card className="mt-4">
          <CardContent className='h-[300px]'>
            <ResponsiveContainer width='100%' height='100%'>
              <ScatterChart>
                <CartesianGrid strokeDasharray='3 3' />
                <XAxis dataKey={x as string} type='number' />
                <YAxis dataKey={x as string} type='number' />
                <Tooltip />
                <Legend />
                <Scatter data={data} fill='#8884d8' />
              </ScatterChart> 
            </ResponsiveContainer>
          </CardContent>
        </Card>
      );
    }

    if ((type === 'bar' || type === 'grouped_bar') && hasCol(x) && pickYArray.length >= 1) {
      return (
        <Card className="mt-4">
          <CardContent className='h-[300px]'>
            <ResponsiveContainer width='100%' height='100%'>
              <BarChart data={data}>
                <CartesianGrid strokeDasharray='3 3' />
                <XAxis dataKey={x as string} interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Legend />
                {pickYArray.map((yk, i) => (
                  <Bar key={yk} dataKey={yk} stroke={COLORS[i % COLORS.length]} />
                ))}
              </BarChart> 
            </ResponsiveContainer>
          </CardContent>
        </Card>
      );
    }

    if (type === 'pie') {
      const nameKey = hasCol(x) ? (x as string) : columns[0];
      const valueKey = pickYArray[0] || columns.find((c) => c !== nameKey) || columns[0];
      return (
        <Card className="mt-4">
          <CardContent className='h-[300px]'>
            <ResponsiveContainer width='100%' height='100%'>
              <PieChart>
                <Pie data={data} dataKey={valueKey} nameKey={nameKey} outerRadius={100} label>
                  {data.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
              </PieChart> 
            </ResponsiveContainer>
          </CardContent>
        </Card>
      );
    }

    // table or unsupported -> fall through to null (table already visible above)
    return null;
  }
  
  // Analyze column types
  const columnTypes = columns.map(col => ({
    name: col,
    type: getColumnType(col, data),
    cardinality: getCardinality(col, data)
  }));

  const numericCols = columnTypes.filter(c => c.type === 'numeric');
  const dateCols = columnTypes.filter(c => c.type === 'date');
  const categoricalCols = columnTypes.filter(c => c.type === 'categorical');
  
  // Find best metric column (prefer by name, then first numeric)
  const preferredMetric = numericCols.find(c => colNameLooksMetric(c.name))?.name || numericCols[0]?.name;

  // 1) TIME SERIES: Date + Numeric → Line Chart
  if (dateCols.length >= 1 && numericCols.length >= 1) {
    const xKey = dateCols[0].name;
    const yKey = preferredMetric!;
    
    const sorted = [...data].sort((a,b) => {
      const ax = new Date(a[xKey]).getTime();
      const bx = new Date(b[xKey]).getTime();
      return ax - bx;
    });
    
    return (
      <Card className="mt-4">
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sorted}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={xKey} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey={yKey} stroke="#5B9BD5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  }

  // 2) SCATTER: Two numeric columns (no categories/dates) → Scatter Plot
  if (numericCols.length >= 2 && categoricalCols.length === 0 && dateCols.length === 0) {
    const [xKey, yKey] = numericCols.slice(0, 2).map(c => c.name);
    
    return (
      <Card className="mt-4">
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={xKey} type="number" />
              <YAxis dataKey={yKey} type="number" />
              <Tooltip />
              <Legend />
              <Scatter data={data} fill="#8884d8" />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  }

  // 3) SINGLE CATEGORY + MULTIPLE METRICS → Grouped Bar
  if (categoricalCols.length === 1 && numericCols.length > 1) {
    const xKey = categoricalCols[0].name;
    const metrics = [preferredMetric!, ...numericCols.filter(c => c.name !== preferredMetric).map(c => c.name)].slice(0, 4);
    
    const prepared = topNWithOthers(
      [...data].sort((a,b) => Number(b[metrics[0]]) - Number(a[metrics[0]])),
      metrics[0],
      12,
      xKey
    );
    
    return (
      <Card className="mt-4">
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={prepared}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={xKey} interval={0} angle={-20} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Legend />
              {metrics.map((m, i) => (
                <Bar key={m} dataKey={m} fill={COLORS[i % COLORS.length]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  }

  // 4) SINGLE CATEGORY + SINGLE METRIC → Smart chart selection
  if (categoricalCols.length === 1 && numericCols.length >= 1) {
    const nameKey = categoricalCols[0].name;
    const valueKey = preferredMetric!;
    const cardinality = categoricalCols[0].cardinality;
    
    const sorted = [...data].sort((a,b) => Number(b[valueKey]) - Number(a[valueKey]));
    const prepared = topNWithOthers(sorted, valueKey, 15, nameKey);
    
    const allPositive = prepared.every(d => Number(d[valueKey]) >= 0);
    const maxLabelLength = Math.max(...prepared.map(d => String(d[nameKey]).length));
    
    // Chart selection logic
    const usePie = prepared.length <= 5 && allPositive;
    const useHorizontalBar = prepared.length > 8 || maxLabelLength > 15;

    return (
      <Card className="mt-4">
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            {usePie ? (
              <PieChart>
                <Pie data={prepared} dataKey={valueKey} nameKey={nameKey} outerRadius={100} label>
                  {prepared.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            ) : useHorizontalBar ? (
              <BarChart data={prepared} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey={nameKey} type="category" width={100} />
                <Tooltip />
                <Legend />
                <Bar dataKey={valueKey} fill="#8884d8" />
              </BarChart>
            ) : (
              <BarChart data={prepared}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={nameKey} interval={0} angle={-45} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey={valueKey} fill="#8884d8" />
              </BarChart>
            )}
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  }

  // 5) SINGLE CATEGORY ONLY → Frequency Chart
  if (categoricalCols.length === 1 && numericCols.length === 0) {
    const nameKey = categoricalCols[0].name;
    const cardinality = categoricalCols[0].cardinality;
    
    const freqMap: Record<string, number> = {};
    data.forEach((row) => {
      const key = row[nameKey] === null ? "NULL" : String(row[nameKey]);
      freqMap[key] = (freqMap[key] || 0) + 1;
    });
    
    const freqData = Object.entries(freqMap).map(([k, v]) => ({ [nameKey]: k, Count: v }));
    const sorted = freqData.sort((a,b) => Number(b.Count) - Number(a.Count));
    const prepared = topNWithOthers(sorted, "Count", 15, nameKey);
    
    const usePie = cardinality <= 6 && prepared.length <= 6;

    return (
      <Card className="mt-4">
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            {usePie ? (
              <PieChart>
                <Pie data={prepared} dataKey="Count" nameKey={nameKey} outerRadius={100} label>
                  {prepared.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            ) : (
              <BarChart data={prepared}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={nameKey} interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Count" fill="#8884d8" />
              </BarChart>
            )}
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  }

  // 6) MULTIPLE CATEGORIES → Show first category as frequency
  if (categoricalCols.length > 1) {
    const nameKey = categoricalCols[0].name; // Use first categorical column
    
    const freqMap: Record<string, number> = {};
    data.forEach((row) => {
      const key = row[nameKey] === null ? "NULL" : String(row[nameKey]);
      freqMap[key] = (freqMap[key] || 0) + 1;
    });
    
    const freqData = Object.entries(freqMap).map(([k, v]) => ({ [nameKey]: k, Count: v }));
    const sorted = freqData.sort((a,b) => Number(b.Count) - Number(a.Count));
    const prepared = topNWithOthers(sorted, "Count", 12, nameKey);

    return (
      <Card className="mt-4">
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={prepared}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={nameKey} interval={0} angle={-20} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Count" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  }

  return null;
};