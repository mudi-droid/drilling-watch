'use client';
import React from 'react';
import { useParams } from 'react-router-dom';
import { v4 as uuid } from 'uuid';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Chat,
  useChatScroll,
  useChatContext,
  ChatMessages,
  ChatProgress,
  ChatTextInput,
  ChatError,
  ChatMessageMemo,
  StepEvent,
  ThinkingEvent,
  ChatProvider,
  StartNewChat,
} from '@/components/block/chat';
import {
  isErrorStateEvent,
  isMessageStateEvent,
  isStepStateEvent,
  isThinkingEvent,
} from '@/components/block/chat/types';
import { type MessageResponse } from '@/api/chat/types';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useMainLayout } from '@/components/block/chat/main-layout-context';
import { FleetProvider, useFleet } from '@/api/fleet/fleet-context';
import { FleetMap } from '@/components/custom/FleetMap';

const initialMessages: MessageResponse[] = [
  {
    id: uuid(),
    role: 'assistant',
    content: {
      format: 2,
      parts: [
        {
          type: 'text',
          text: `I'm the Watcher for the four-well fleet on the map. Ask "what's the fleet doing?" to score every well, or ask about a single well or a moment in the hour.`,
        },
      ],
    },
    createdAt: new Date(),
    type: 'initial',
  },
];

export interface ChatPageContentProps {
  chatId: string;
  hasChat: boolean;
  isNewChat: boolean;
  isLoadingChats: boolean;
  addChatHandler: () => void;
}

export function ChatImplementation({ chatId }: { chatId: string }) {
  const {
    sendMessage,
    userInput,
    setUserInput,
    combinedEvents,
    progress,
    deleteProgress,
    isLoadingHistory,
    isAgentRunning,
  } = useChatContext();

  const { scrollContainerRef, onChatScroll } = useChatScroll({
    chatId,
    events: combinedEvents,
  });

  // Example for a tool with a handler
  // useAgUiTool({
  //   name: 'alert',
  //   description: 'Action. Display an alert to the user',
  //   handler: ({ message }) => alert(message),
  //   parameters: z.object({
  //     message: z
  //       .string()
  //       .describe('The message that will be displayed to the user'),
  //   }),
  //   background: false,
  // });
  //
  // Example for a custom UI widget
  //
  // useAgUiTool({
  //   name: 'weather',
  //   description: 'Widget. Displays weather result to user',
  //   render: ({ args }) => {
  //     return <WeatherWidget {...args} />;
  //   },
  //   parameters: z.object({
  //     temperature: z.number(),
  //     feelsLike: z.number(),
  //     humidity: z.number(),
  //     windSpeed: z.number(),
  //     windGust: z.number(),
  //     conditions: z.string(),
  //     location: z.string(),
  //   }),
  // });

  return (
    <div className="flex h-svh w-full min-w-0 flex-col">
      {/* Fleet map on top with an EXPLICIT height - Leaflet renders nothing in a
          zero-height parent. Drawn on first paint, before any tool has run. */}
      <div className="h-[38vh] min-h-[260px] w-full min-w-0 shrink-0 border-b border-border">
        <FleetMap />
      </div>

      {/* Chat full width beneath, flex-1 min-h-0 so it fills the rest. */}
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col p-4">
        <Chat initialMessages={initialMessages}>
          <ScrollArea
            className="mb-5 min-h-0 w-full min-w-0 flex-1"
            scrollViewportRef={scrollContainerRef}
            onWheel={onChatScroll}
          >
            <div className="w-full min-w-0 justify-self-center">
              <ChatMessages isLoading={isLoadingHistory} messages={combinedEvents} chatId={chatId}>
                {combinedEvents &&
                  combinedEvents.map(m => {
                    if (isErrorStateEvent(m)) {
                      return <ChatError key={m.value.id} {...m.value} />;
                    }
                    if (isMessageStateEvent(m)) {
                      return <ChatMessageMemo key={m.value.id} {...m.value} />;
                    }
                    if (isStepStateEvent(m)) {
                      return <StepEvent key={m.value.id} {...m.value} />;
                    }
                    if (isThinkingEvent(m)) {
                      return <ThinkingEvent key={m.type} />;
                    }
                  })}
              </ChatMessages>
              <ChatProgress progress={progress || {}} deleteProgress={deleteProgress} />
            </div>
          </ScrollArea>

          <ChatTextInput
            userInput={userInput}
            setUserInput={setUserInput}
            onSubmit={sendMessage}
            runningAgent={isAgentRunning}
          />
        </Chat>
      </div>
    </div>
  );
}

export const ChatPage: React.FC = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const { hasChat, isNewChat, isLoadingChats, addChatHandler, refetchChats } = useMainLayout();

  if (!chatId) {
    return null;
  }

  if (isLoadingChats) {
    return (
      <div className="flex w-full flex-1 flex-col space-y-4 p-4">
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-20 w-full" />
      </div>
    );
  }

  if (!hasChat) {
    return <StartNewChat createChat={addChatHandler} />;
  }

  return (
    <FleetProvider>
      <ChatWithFleet chatId={chatId} isNewChat={isNewChat} refetchChats={refetchChats} />
    </FleetProvider>
  );
};

function ChatWithFleet({
  chatId,
  isNewChat,
  refetchChats,
}: {
  chatId: string;
  isNewChat: boolean;
  refetchChats: () => Promise<unknown>;
}) {
  // The subscriber lets this chat feed the fleet map from the tool event stream.
  const { subscriber } = useFleet();
  return (
    <ChatProvider
      chatId={chatId}
      runInBackground={true}
      isNewChat={isNewChat}
      refetchChats={refetchChats}
      subscriber={subscriber}
    >
      <ChatImplementation chatId={chatId} />
    </ChatProvider>
  );
}
