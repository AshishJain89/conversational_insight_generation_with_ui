import { Card, CardContent } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, } from "recharts";

interface ChartRendererProps {
  columns: string[];
  rows: (string | number | null)[][];
}

const COLORS = [ "#8884d8", "#82ca9d", "#ffc658", "#ff7f50", "#00c49f", "#0088fe", "#ffbb28", "#ff4444", ];

export const ChartRenderer = ({ columns, rows }: ChartRendererProps) => {
  if (!rows || rows.length === 0 || columns.length === 0) return null;

  // Transform into recharts-compatible format
  const data = rows.map((row) => {
    const obj: Record<string, any> = {};
    columns.forEach((col, i) => {
      obj[col] = row[i] === null ? "NULL" : row[i];
    });
    return obj;
  });

  // Identify numeric vs categorical columns
  const numericCols = columns.filter((col) =>
    data.every((row) => !isNaN(Number(row[col])))
  );
  const categoricalCols = columns.filter((col) => !numericCols.includes(col));

  // Pick best chart type
  let chart = null;

  if (numericCols.length === 1 && categoricalCols.length >= 1) {
    // Example: ProductName vs TotalSales → BarChart
    const xKey = categoricalCols[0];
    const yKey = numericCols[0];
    chart = (
      <BarChart data={data}>
        <XAxis dataKey={xKey} hide={false} />
        <YAxis />
        <Tooltip />
        <Bar dataKey={yKey} fill="#8884d8" />
      </BarChart>
    );
  } else if (numericCols.length === 2) {
    // Example: Date vs Sales → LineChart
    const [xKey, yKey] = numericCols;
    chart = (
      <LineChart data={data}>
        <XAxis dataKey={xKey} />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey={yKey} stroke="#82ca9d" />
      </LineChart>
    );
  } else if (categoricalCols.length === 1 && numericCols.length === 1) {
    // Example: Country vs Count → PieChart
    const nameKey = categoricalCols[0];
    const valueKey = numericCols[0];
    chart = (
      <PieChart>
        <Pie
          data={data}
          dataKey={valueKey}
          nameKey={nameKey}
          outerRadius={100}
          fill="#8884d8"
          label
        >
          {data.map((_, index) => (
            <Cell key={index} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    );
  }

  if (!chart) return null;

  return (
    <Card className="mt-4">
      <CardContent className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          {chart}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
