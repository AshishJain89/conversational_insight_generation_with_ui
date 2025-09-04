import { Message } from './ChatWindow';
import { cn } from '@/lib/utils';
import { ChartRenderer } from './ChartRenderer';
import { TrendingUp, BarChart3 } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isUser = message.role === 'user';



  const renderContent = () => {
    if (isUser) return <div className="whitespace-pre-wrap">{message.content}</div>;

    switch (message.type) {
      case 'sql':
        return (
          <div className="bg-slate-800 text-slate-100 p-4 rounded-lg font-mono text-sm overflow-x-auto">
            <div className="text-xs text-slate-400 mb-2 uppercase tracking-wide flex items-center gap-2">
              <BarChart3 className="h-3 w-3" />
              SQL Query Generated
            </div>
            <div className="font-mono text-sm">
              {message.content.split(/\b(SELECT|FROM|WHERE|JOIN|AS|SUM|COUNT|LIMIT|GROUP\s+BY|ORDER\s+BY|DESC|ASC|INNER|LEFT|RIGHT|ON)\b/gi).map((part, index) => {
                if (/\b(SELECT|FROM|WHERE|JOIN|AS|SUM|COUNT|LIMIT|GROUP\s+BY|ORDER\s+BY|DESC|ASC|INNER|LEFT|RIGHT|ON)\b/i.test(part)) {
                  return (
                    <span key={index} className="text-green-400 font-bold">
                      {part}
                    </span>
                  );
                }
                return <span key={index}>{part}</span>;
              })}
            </div>
          </div>
        );

      case 'table':
        return (
          <div className="space-y-4">
            <div className="bg-slate-50 dark:bg-slate-700 p-4 rounded-lg overflow-x-auto">
              <div className="text-xs text-green-600 dark:text-green-400 mb-2 uppercase tracking-wide">
                Query Results ({message.rows?.length || 0} rows)
              </div>
              <div className="overflow-x-auto max-w-full">
                <table className="min-w-max text-sm border-collapse border border-slate-400 dark:border-slate-600">
                  <thead>
                    <tr className="bg-slate-100 dark:bg-slate-600">
                      {message.columns?.map((col, idx) => (
                        <th
                          key={idx}
                          className="border px-3 py-2 dark:border-slate-600 text-left whitespace-nowrap font-semibold"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {message.rows?.length ? (
                      message.rows.slice(0, 100).map((row, i) => ( // Limit to first 100 rows for display
                        <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-600">
                          {row.map((cell, j) => (
                            <td
                              key={j}
                              className="border px-3 py-2 dark:border-slate-600 whitespace-nowrap"
                            >
                              {cell === null ? <span className="text-gray-400 italic">NULL</span> : String(cell)}
                            </td>
                          ))}
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td
                          colSpan={message.columns?.length || 1}
                          className="text-center text-muted-foreground py-4"
                        >
                          No rows returned
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              {message.rows && message.rows.length > 100 && (
                <div className="text-xs text-muted-foreground mt-2">
                  Showing first 100 of {message.rows.length} rows
                </div>
              )}
            </div>
          </div>
        );
      case 'chart':
        return (
          <div className="bg-slate-50 dark:bg-slate-700 p-4 rounded-lg overflow-x-auto">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-2 uppercase tracking-wide">
              Chart
            </div>
            <ChartRenderer columns={message.columns || []} rows={message.rows || []} suggestion={message.chart} />
          </div>
        );
      case 'forecast':
        return (
          <div className="space-y-4">
            {/* Forecast intent message */}
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-700">
              <div className="text-xs text-yellow-700 dark:text-yellow-400 mb-2 uppercase tracking-wide flex items-center gap-2">
                <TrendingUp className="h-3 w-3" />
                Time Series Data Retrieved
              </div>
              <div className="text-sm text-yellow-800 dark:text-yellow-200">
                {message.content}
              </div>
              <div className="mt-2 text-xs text-yellow-600 dark:text-yellow-300">
                💡 This data can be used for forecasting. Click "Generate Forecast" to predict future values.
              </div>
            </div>
            
            {/* Data table - same as table case */}
            {message.columns && message.rows && (
              <div className="bg-slate-50 dark:bg-slate-700 p-4 rounded-lg overflow-x-auto">
                <div className="text-xs text-green-600 dark:text-green-400 mb-2 uppercase tracking-wide">
                  Query Results ({message.rows.length} rows)
                </div>
                <div className="overflow-x-auto max-w-full">
                  <table className="min-w-max text-sm border-collapse border border-slate-400 dark:border-slate-600">
                    <thead>
                      <tr className="bg-slate-100 dark:bg-slate-600">
                        {message.columns.map((col, idx) => (
                          <th
                            key={idx}
                            className="border px-3 py-2 dark:border-slate-600 text-left whitespace-nowrap font-semibold"
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {message.rows.length ? (
                        message.rows.slice(0, 100).map((row, i) => (
                          <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-600">
                            {row.map((cell, j) => (
                              <td
                                key={j}
                                className="border px-3 py-2 dark:border-slate-600 whitespace-nowrap"
                              >
                                {cell === null ? <span className="text-gray-400 italic">NULL</span> : String(cell)}
                              </td>
                            ))}
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td
                            colSpan={message.columns.length || 1}
                            className="text-center text-muted-foreground py-4"
                          >
                            No rows returned
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                {message.rows.length > 100 && (
                  <div className="text-xs text-muted-foreground mt-2">
                    Showing first 100 of {message.rows.length} rows
                  </div>
                )}
              </div>
            )}
          </div>
        );

      case 'forecast_result':
        const forecast = message.forecastData;
        return (
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-700">
              <div className="text-xs text-blue-700 dark:text-blue-400 mb-3 uppercase tracking-wide flex items-center gap-2">
                <TrendingUp className="h-3 w-3" />
                ARIMA Forecast Results
              </div>
              
              {forecast?.error ? (
                <div className="text-red-600 dark:text-red-400 text-sm">
                  Error: {forecast.error}
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Model Info */}
                  {forecast?.model && (
                    <div className="text-sm">
                      <span className="font-medium">Model:</span> ARIMA
                      {forecast.model.order && ` (${forecast.model.order.join(',')})`}
                      {forecast.model.seasonal_order && forecast.model.seasonal_order.some((x: number) => x > 0) && 
                        ` × (${forecast.model.seasonal_order.join(',')})`}
                    </div>
                  )}
                  
                  {/* Diagnostics */}
                  {forecast?.diagnostics && (
                    <div className="text-xs space-y-1 text-muted-foreground">
                      {forecast.diagnostics.aic && <div>AIC: {forecast.diagnostics.aic.toFixed(2)}</div>}
                      {forecast.diagnostics.rmse && <div>RMSE: {forecast.diagnostics.rmse.toFixed(4)}</div>}
                    </div>
                  )}
                  
                  {/* Forecast Plot */}
                  {forecast?.plot_png_base64 && (
                    <div className="mt-4">
                      <img 
                        src={forecast.plot_png_base64} 
                        alt="Forecast Plot" 
                        className="w-full rounded-lg border border-border/20 shadow-sm"
                      />
                    </div>
                  )}
                  
                  {/* Forecast Table */}
                  {forecast?.forecast && forecast.forecast.length > 0 && (
                    <details className="mt-4">
                      <summary className="cursor-pointer text-sm font-medium text-blue-700 dark:text-blue-400 hover:underline">
                        Show Forecast Data ({forecast.forecast.length} points)
                      </summary>
                      <div className="mt-2 overflow-x-auto">
                        <table className="min-w-full text-xs border-collapse border border-slate-300 dark:border-slate-600">
                          <thead>
                            <tr className="bg-slate-100 dark:bg-slate-700">
                              <th className="border px-2 py-1 dark:border-slate-600">Date</th>
                              <th className="border px-2 py-1 dark:border-slate-600">Forecast</th>
                              <th className="border px-2 py-1 dark:border-slate-600">Lower CI</th>
                              <th className="border px-2 py-1 dark:border-slate-600">Upper CI</th>
                            </tr>
                          </thead>
                          <tbody>
                            {forecast.forecast.slice(0, 12).map((point: any, i: number) => (
                              <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-600">
                                <td className="border px-2 py-1 dark:border-slate-600">{point.date}</td>
                                <td className="border px-2 py-1 dark:border-slate-600">{point.mean.toFixed(2)}</td>
                                <td className="border px-2 py-1 dark:border-slate-600">{point.lower.toFixed(2)}</td>
                                <td className="border px-2 py-1 dark:border-slate-600">{point.upper.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {forecast.forecast.length > 12 && (
                          <div className="text-xs text-muted-foreground mt-1">
                            Showing first 12 of {forecast.forecast.length} forecast points
                          </div>
                        )}
                      </div>
                    </details>
                  )}
                </div>
              )}
            </div>
          </div>
        );

      default:
        return <div className="whitespace-pre-wrap">{message.content}</div>;
    }
  };

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[80%] rounded-2xl px-4 py-3 shadow-sm transition-all duration-300 hover:scale-[1.02]',
          isUser
            ? 'bg-primary text-primary-foreground ml-12'
            : 'bg-chat-surface text-foreground mr-12 border border-border/20'
        )}
      >
        {renderContent()}
        <div
          className={cn(
            'text-xs mt-2 opacity-70',
            isUser ? 'text-primary-foreground/70' : 'text-muted-foreground'
          )}
        >
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  );
};