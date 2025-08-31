import { Message } from './ChatWindow';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isUser = message.role === 'user';
  
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
        <div className="whitespace-pre-wrap break-words">
          {message.content}
        </div>
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