import { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { TrendingUp, Database } from 'lucide-react';

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  type?: 'text' | 'sql' | 'table' | 'chart' | 'insight' | 'forecast' | 'forecast_result';
  columns?: string[];
  rows?: any[][];
  sql?: string;
  forecastData?: any;
  insight?: string;
  chart?: {
    type: 'line' | 'bar' | 'pie' | 'grouped_bar' | 'scatter' | 'table';
    x?: string | null;
    y?: string | string[] | null;
    series?: string | null;
    reason?: string | null;
  }
}

export const ChatWindow = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [forecastLoading, setForecastLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => { scrollToBottom(); }, [messages]);

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: 'user',
      timestamp: new Date(),
      type: 'text',
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/nl2sql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: content, execute: true }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      console.log('Backend response:', data);
      console.log('Chart suggestion:', data.chart);
      
      // Create the 4-message sequence: SQL → Table → Chart → Insight
      const newMessages: Message[] = [];
      let messageIdCounter = Date.now();
      
      // 1. SQL Message (always show if SQL was generated)
      if (data.sql) {
        const sqlMessage: Message = {
          id: (++messageIdCounter).toString(),
          content: data.sql,
          role: 'assistant',
          timestamp: new Date(),
          type: 'sql',
        };
        newMessages.push(sqlMessage);
      }

      // 2. Table Message (show if we have data)
      if (data.columns && data.rows) {
        const tableMessage: Message = {
          id: (++messageIdCounter).toString(),
          content: data.forecast ? 
            `Retrieved ${data.rows.length} rows for forecasting analysis.` : 
            "Here are the results:",
          role: 'assistant',
          timestamp: new Date(),
          type: data.forecast ? 'forecast' : 'table',
          columns: data.columns,
          rows: data.rows,
          sql: data.sql,
        };
        newMessages.push(tableMessage);

        // 3, Chart Message (show if chart suggestion exists and there are rows)
        if (data.chart && data.rows && data.rows.length > 0) {
          const chartMessage: Message = {
            id: (++messageIdCounter).toString(),
            content: '', // Chart doesn't need text content
            role: 'assistant',
            timestamp: new Date(),
            type: 'chart',
            columns: data.columns,
            rows: data.rows,
            chart: data.chart,
          };
          newMessages.push(chartMessage);
        }

        // 4. Insight Message (show if insight was generated)
        if (data.insight) {
          const insightMessage: Message = {
            id: (++messageIdCounter).toString(),
            content: data.insight,
            role: 'assistant',
            timestamp: new Date(),
            type: 'insight',
            insight: data.insight,
          };
          newMessages.push(insightMessage)
        }
      } else if (data.message) {
        // Error or text response
        const textMessage: Message = {
          id: (++messageIdCounter).toString(),
          content: data.message,
          role: 'assistant',
          timestamp: new Date(),
          type: 'text',
        };
        newMessages.push(textMessage);
      }

      // Add messages with small delays for better UX
      for (let i = 0; i < newMessages.length; i++) {
        setTimeout(() => {
          setMessages(prev => {
            // Only add if this message isn't already added
            if (prev.find(m => m.id === newMessages[i].id)) return prev;
            return [...prev, newMessages[i]];
          });
        }, i * 500); // 500ms delay between messages
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast({
        title: "Error",
        description: "Failed to get response from the server. Please try again.",
        variant: "destructive",
      });

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "Sorry, I couldn't process your request. Please make sure the server is running and try again.",
        role: 'assistant',
        timestamp: new Date(),
        type: 'text',
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const runForecast = async (sql: string, columns: string[], rows: any[][]) => {
    setForecastLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          columns,
          rows,
          horizon: 12,
          seasonal: true,
          missing_method: 'interpolate'
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const forecastData = await response.json();

      const forecastMessage: Message = {
        id: Date.now().toString(),
        content: "Forecast Analysis Complete",
        role: 'assistant',
        timestamp: new Date(),
        type: 'forecast_result',
        forecastData,
      };

      setMessages(prev => [...prev, forecastMessage]);
      
      toast({
        title: "Forecast Complete",
        description: `Generated ${forecastData.forecast?.length || 0} forecast points`,
      });

    } catch (error) {
      console.error('Forecast error:', error);
      toast({
        title: "Forecast Error",
        description: "Failed to generate forecast. Please try again.",
        variant: "destructive",
      });

      const errorMessage: Message = {
        id: Date.now().toString(),
        content: "Failed to generate forecast. The data might not be suitable for time series analysis.",
        role: 'assistant',
        timestamp: new Date(),
        type: 'text',
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setForecastLoading(false);
    }
  };

  // Add forecast button to messages that have forecast-ready data
  const renderForecastButton = (message: Message) => {
    if ((message.type === 'table' || message.type === 'forecast') && message.sql && message.columns && message.rows) {
      const hasDateColumn = message.columns.some(col => 
        col.toLowerCase().includes('date') || 
        col.toLowerCase().includes('time') ||
        col.toLowerCase().includes('month') ||
        col.toLowerCase().includes('year')
      );
      
      const hasNumericColumn = message.columns.some(col => 
        col.toLowerCase().includes('sales') ||
        col.toLowerCase().includes('amount') ||
        col.toLowerCase().includes('revenue') ||
        col.toLowerCase().includes('quantity') ||
        col.toLowerCase().includes('count')
      );

      if (hasDateColumn && hasNumericColumn && message.rows && message.rows.length > 3) {
        return (
          <div className="mt-3 pt-3 border-t border-border/20">
            <div className="text-xs text-muted-foreground mb-2">
              💡 This appears to be time-series data. You can analyze historical trends or generate future forecasts.
            </div>
            <Button
              onClick={() => runForecast(message.sql!, message.columns!, message.rows!)}
              disabled={forecastLoading}
              variant="secondary"
              size="sm"
              className="gap-2"
            >
              <TrendingUp className="h-4 w-4" />
              {forecastLoading ? 'Generating Forecast...' : 'Generate Forecast'}
            </Button>
          </div>
        );
      }
    }
    return null;
  };

  return (
    <div className="flex flex-col h-screen bg-chat-bg">
      <div className="flex-shrink-0 bg-chat-surface/90 backdrop-blur-md border-b border-border/50 p-4 shadow-sm">
        <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
          Conversational Insight Generation Bot
        </h1>
        <p className="text-muted-foreground">Ask questions and get SQL queries instantly</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="text-6xl mb-4">💬</div>
              <h2 className="text-xl font-semibold mb-2 text-foreground/80">Hello! I'm your Insight Assistant</h2>
              <p className="text-muted-foreground">Ask questions about the Northwind database to get started!</p>
              <div className="mt-4 space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-2 justify-center">
                  <Database className="h-4 w-4" />
                  <span>Try: "Show me top selling products"</span>
                </div>
                <div className="flex items-center gap-2 justify-center">
                  <TrendingUp className="h-4 w-4" />
                  <span>Try: "Forecast monthly sales for next year"</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id}>
            <ChatMessage message={message} />
            {renderForecastButton(message)}
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-chat-assistant rounded-2xl px-4 py-3 max-w-sm">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="flex-shrink-0 p-4 bg-chat-surface/90 backdrop-blur-md border-t border-border/50">
        <ChatInput onSendMessage={sendMessage} disabled={isLoading} />
      </div>
    </div>
  );
};