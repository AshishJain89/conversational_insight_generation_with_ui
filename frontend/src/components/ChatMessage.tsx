import { Message } from './ChatWindow';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isUser = message.role === 'user';
  
  // Check if assistant message contains SQL
  const isSQL = !isUser && (message.content.includes('SELECT') || message.content.includes('sql'));
  
  const formatSQL = (content: string) => {
    try {
      const data = JSON.parse(content);
  
      if (data.sql) {
        return (
          <div className="space-y-4">
            {/* SQL Query */}
            <div className="bg-slate-800 text-slate-100 p-4 rounded-lg font-mono text-sm overflow-x-auto">
              <div className="text-xs text-slate-400 mb-2 uppercase tracking-wide">SQL Query</div>
              <div
                dangerouslySetInnerHTML={{
                  __html: data.sql
                    .replace(/\b(SELECT|FROM|WHERE|JOIN|AS|SUM|COUNT)\b/g, '<span class="text-green-400 font-bold">$1</span>')
                    .replace(/'([^']*)'/g, '<span class="text-orange-300">\'$1\'</span>')
                }}
              />
            </div>
  
            {/* Results */}
            {(data.table || data.rows) && (
              <div className="bg-slate-50 dark:bg-slate-700 p-4 rounded-lg">
                <div className="text-xs text-green-600 dark:text-green-400 mb-2 uppercase tracking-wide flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                  Query Results
                </div>
                {data.table ? (
                  <pre className="font-mono text-sm whitespace-pre-wrap">{data.table}</pre>
                ) : (
                  <table className="text-sm font-mono border-collapse">
                    <thead>
                      <tr>
                        {data.columns?.map((col: string) => (
                          <th key={col} className="border px-2 py-1">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {data.rows?.map((row: any[], i: number) => (
                        <tr key={i}>
                          {row.map((cell, j) => (
                            <td key={j} className="border px-2 py-1">{cell}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </div>
        );
      }
    } catch (e) {
      // fallback for plain SQL strings
      if (content.includes('SELECT') || content.includes('FROM')) {
        return (
          <div className="bg-slate-800 text-slate-100 p-4 rounded-lg font-mono text-sm overflow-x-auto">
            <div
              dangerouslySetInnerHTML={{
                __html: content.replace(/\b(SELECT|FROM|WHERE|JOIN|AS|SUM|COUNT)\b/g, '<span class="text-green-400 font-bold">$1</span>')
                  .replace(/'([^']*)'/g, '<span class="text-orange-300">\'$1\'</span>')
              }}
            />
          </div>
        );
      }
    }
  
    return content;
  };
  
  
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3 shadow-sm transition-all duration-300 hover:scale-[1.02]",
          isUser
            ? "bg-primary text-primary-foreground ml-12"
            : "bg-chat-surface text-foreground mr-12 border border-border/20"
        )}
      >
        {isSQL ? formatSQL(message.content) : message.content}
        <div
          className={cn(
            "text-xs mt-2 opacity-70",
            isUser ? "text-primary-foreground/70" : "text-muted-foreground"
          )}
        >
          {message.timestamp.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
          })}
        </div>
      </div>
    </div>
  );
};