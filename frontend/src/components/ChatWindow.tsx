import { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { useToast } from '@/hooks/use-toast';

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export const ChatWindow = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/nl2sql', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: content, execute: true }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: JSON.stringify({
          sql: data.sql || 'No SQL generated',
          // Use:
          result: data.rows && data.rows.length > 0 ? 
          `Columns: ${data.columns.join(', ')}\nData:\n${data.rows.map(row => row.join(', ')).join('\n')}` : 
          'No data found'
        }),
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
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
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-chat-bg">
      {/* Header */}
      <div className="flex-shrink-0 bg-chat-surface/90 backdrop-blur-md border-b border-border/50 p-4 shadow-sm">
        <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
          SQL Chat Assistant
        </h1>
        <p className="text-muted-foreground">Ask questions and get SQL queries instantly</p>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="text-6xl mb-4">💬</div>
              <h2 className="text-xl font-semibold mb-2 text-foreground/80">Welcome to SQL Chat</h2>
              <p className="text-muted-foreground">Type your question below to get started!</p>
            </div>
          </div>
        )}
        
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
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

      {/* Input */}
      <div className="flex-shrink-0 p-4 bg-chat-surface/90 backdrop-blur-md border-t border-border/50">
        <ChatInput onSendMessage={sendMessage} disabled={isLoading} />
      </div>
    </div>
  );
};