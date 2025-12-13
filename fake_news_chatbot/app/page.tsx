'use client';

import { useState, useCallback } from 'react';
import { ChatSidebar } from './components/ChatSidebar';
import { ChatInterface } from './components/ChatInterface';
import { ThemeToggle } from './components/ThemeToggle';

interface Message {
  id: string;
  text: string;
  label: 'FAKE' | 'REAL';
  confidence: number;
  timestamp: Date;
}

interface Chat {
  id: string;
  title: string;
  timestamp: Date;
  preview: string;
  messages: Message[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000';

export default function Home() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const activeChat = chats.find((chat) => chat.id === activeChatId);
  const activeMessages = activeChat?.messages || [];

  const generateChatId = () => {
    return `chat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  };

  const generateMessageId = () => {
    return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  };

  const createNewChat = useCallback(() => {
    const newChat: Chat = {
      id: generateChatId(),
      title: 'New Chat',
      timestamp: new Date(),
      preview: 'Start a new conversation...',
      messages: [],
    };
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
  }, []);

  const handleSendMessage = useCallback(
    async (text: string) => {
      if (!activeChatId) {
        createNewChat();
        // Wait a bit for state to update
        setTimeout(() => handleSendMessage(text), 100);
        return;
      }

      setIsLoading(true);

      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text }),
        });

        if (!response.ok) {
          throw new Error('Failed to get prediction');
        }

        const data = await response.json();

        const newMessage: Message = {
          id: generateMessageId(),
          text,
          label: data.label as 'FAKE' | 'REAL',
          confidence: data.confidence,
          timestamp: new Date(),
        };

        setChats((prev) =>
          prev.map((chat) => {
            if (chat.id === activeChatId) {
              const updatedMessages = [...chat.messages, newMessage];
              return {
                ...chat,
                title: text.length > 30 ? text.substring(0, 30) + '...' : text,
                preview: text.length > 50 ? text.substring(0, 50) + '...' : text,
                messages: updatedMessages,
                timestamp: new Date(),
              };
            }
            return chat;
          })
        );
      } catch (error) {
        console.error('Error sending message:', error);
        alert('Failed to analyze news. Please make sure the Flask server is running.');
      } finally {
        setIsLoading(false);
      }
    },
    [activeChatId, createNewChat]
  );

  const handleSelectChat = useCallback((id: string) => {
    setActiveChatId(id);
  }, []);

  const handleDeleteChat = useCallback((id: string) => {
    setChats((prev) => prev.filter((chat) => chat.id !== id));
    if (activeChatId === id) {
      setActiveChatId(null);
      if (chats.length > 1) {
        const remainingChats = chats.filter((chat) => chat.id !== id);
        setActiveChatId(remainingChats[0]?.id || null);
      }
    }
  }, [activeChatId, chats]);

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      {/* Sidebar */}
      <div className="w-80 flex-shrink-0 border-r border-gray-200 dark:border-gray-800">
        <ChatSidebar
          chats={chats}
          activeChatId={activeChatId}
          onSelectChat={handleSelectChat}
          onNewChat={createNewChat}
          onDeleteChat={handleDeleteChat}
        />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative">
        {/* Header with Theme Toggle */}
        <div className="absolute top-4 right-4 z-10">
          <ThemeToggle />
        </div>

        {/* Chat Interface */}
        <div className="flex-1 overflow-hidden">
          {activeChatId ? (
            <ChatInterface
              messages={activeMessages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
            />
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  Welcome to Fake News Detection
                </h2>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Click "New Chat" to start analyzing news articles
                </p>
                <button
                  onClick={createNewChat}
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white rounded-lg font-semibold transition-all duration-200 shadow-lg hover:shadow-xl"
                >
                  Start New Chat
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

