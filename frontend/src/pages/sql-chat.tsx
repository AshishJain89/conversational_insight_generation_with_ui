import React, { useState } from 'react';
import './sql-chat.css';

interface Message {
  id: number;
  type: 'user' | 'sql' | 'result';
  content: string;
  sql?: string;
  result?: any;
  status?: 'success' | 'error';
  executionTime?: string;
}

const SqlChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      type: 'user',
      content: 'Total sales of 2024?'
    },
    {
      id: 2,
      type: 'sql',
      content: 'SQL Query Generated',
      sql: `SELECT SUM(o.quantity * p.price) AS total_sales_2024
FROM orders o
JOIN products p ON o.product_id = p.id
JOIN customers c ON o.customer_id = c.id
WHERE STRFTIME('%Y', o.order_date) = '2024';`
    },
    {
      id: 3,
      type: 'result',
      content: 'Query Results',
      result: { total_sales_2024: '$2,847,390.50' },
      status: 'success',
      executionTime: '0.12s'
    }
  ]);
  
  const [inputValue, setInputValue] = useState('');

  const handleSendMessage = () => {
    if (inputValue.trim()) {
      const newMessage: Message = {
        id: messages.length + 1,
        type: 'user',
        content: inputValue
      };
      
      setMessages(prev => [...prev, newMessage]);
      setInputValue('');
      
      // Simulate SQL response
      setTimeout(() => {
        const sqlMessage: Message = {
          id: messages.length + 2,
          type: 'sql',
          content: 'SQL Query Generated',
          sql: `SELECT * FROM your_table 
WHERE condition = '${inputValue}';`
        };
        
        setMessages(prev => [...prev, sqlMessage]);
        
        // Simulate result
        setTimeout(() => {
          const resultMessage: Message = {
            id: messages.length + 3,
            type: 'result',
            content: 'Query Results',
            result: { message: 'Results would appear here based on your database' },
            status: 'success',
            executionTime: '0.08s'
          };
          
          setMessages(prev => [...prev, resultMessage]);
        }, 800);
      }, 500);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  const formatSQL = (sql: string) => {
    return sql
      .replace(/\b(SELECT|FROM|JOIN|WHERE|ON|AS|SUM|STRFTIME)\b/g, '<span class="sql-keyword">$1</span>')
      .replace(/'([^']*)'/g, '<span class="sql-string">\'$1\'</span>')
      .replace(/\b(SUM|STRFTIME)\b/g, '<span class="sql-function">$1</span>');
  };

  return (
    <div className="sql-chat-body">
      <div className="header">
        <h1>SQL Chat Assistant</h1>
        <p>Ask questions and get SQL queries instantly</p>
      </div>

      <div className="chat-container">
        {messages.map((message) => (
          <div key={message.id}>
            {message.type === 'user' && (
              <div className="message user-message">
                <strong>You:</strong> {message.content}
              </div>
            )}
            
            {message.type === 'sql' && (
              <div className="message assistant-message">
                <div className="message-header">
                  <span>🔍</span>
                  {message.content}
                </div>
                <div className="sql-block">
                  <div dangerouslySetInnerHTML={{ __html: formatSQL(message.sql || '') }} />
                </div>
              </div>
            )}
            
            {message.type === 'result' && (
              <div className="message assistant-message">
                <div className="message-header">
                  <span>📊</span>
                  {message.content}
                </div>
                <div className="result-section">
                  <div className="result-header">
                    <span className={`status-badge ${message.status === 'error' ? 'error-badge' : ''}`}>
                      {message.status}
                    </span>
                    <span>Query executed in {message.executionTime}</span>
                  </div>
                  
                  {message.result && typeof message.result === 'object' && message.result.total_sales_2024 ? (
                    <div className="metric-card">
                      <div className="metric-value">{message.result.total_sales_2024}</div>
                      <div className="metric-label">Total Sales 2024</div>
                    </div>
                  ) : (
                    <div className="no-data">
                      {message.result?.message || 'No data returned'}
                    </div>
                  )}
                  
                  <table className="result-table" style={{ marginTop: '20px' }}>
                    <thead>
                      <tr>
                        {Object.keys(message.result || {}).map(key => (
                          <th key={key}>{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        {Object.values(message.result || {}).map((value, idx) => (
                          <td key={idx}>{String(value)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="input-container">
        <input
          type="text"
          className="input-field"
          placeholder="Ask me anything about SQL queries..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button className="send-btn" onClick={handleSendMessage}>
          Send
        </button>
      </div>
    </div>
  );
};

export default SqlChat;