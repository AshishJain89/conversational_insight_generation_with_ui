import { useState, KeyboardEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export const ChatInput = ({ onSendMessage, disabled }: ChatInputProps) => {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex gap-3 items-end">
      <div className="flex-1 relative">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask me anything about SQL queries..."
          disabled={disabled}
          className={cn(
            "min-h-[60px] max-h-[120px] resize-none pr-12 bg-chat-input-bg border-border/30",
            "focus:border-primary/50 focus:ring-2 focus:ring-primary/20 transition-all duration-200",
            "placeholder:text-muted-foreground/60"
          )}
          rows={1}
        />
      </div>
      <Button
        onClick={handleSend}
        disabled={disabled || !input.trim()}
        size="lg"
        className={cn(
          "h-[60px] w-[60px] p-0 bg-gradient-primary hover:scale-105 transition-all duration-200",
          "disabled:opacity-50 disabled:hover:scale-100 shadow-lg hover:shadow-xl"
        )}
      >
        <Send className="h-5 w-5" />
      </Button>
    </div>
  );
};