import { useLayoutEffect } from 'react';
import { Outlet, useNavigate, useParams, useMatch } from 'react-router-dom';
import { ChatSidebar } from '@/components/block/chat/chat-sidebar';
import { useChatList } from '@/components/block/chat/hooks/use-chat-list';
import { MainLayoutProvider } from '@/components/block/chat/main-layout-context';

export function MainLayout() {
  const { chatId = '' } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();

  const setChatIdHandler = (id: string) => {
    navigate(`/chat/${id}`);
  };

  const isChatEmptyPage = useMatch('/chat');
  const isChatSelectedPage = useMatch('/chat/:chatId');
  const isChat = isChatEmptyPage || isChatSelectedPage;

  const {
    hasChat,
    isNewChat,
    chats,
    isLoadingChats,
    addChatHandler,
    deleteChatHandler,
    isDeletingChat,
    refetchChats,
  } = useChatList({
    chatId,
    setChatId: setChatIdHandler,
    showStartChat: !chatId,
  });

  useLayoutEffect(() => {
    if (isLoadingChats || !chats || chats?.find(c => c.id === chatId)) {
      return;
    }
    if (!isChat) {
      return;
    }
    if (!chats.length) {
      addChatHandler();
    } else {
      setChatIdHandler(chats[0].id);
    }
  }, [chats, isLoadingChats, isChat, chatId]);

  return (
    <div className="flex h-svh w-full flex-row">
      <ChatSidebar
        isLoading={isLoadingChats}
        chatId={chatId}
        chats={chats}
        onChatCreate={addChatHandler}
        onChatSelect={setChatIdHandler}
        onChatDelete={deleteChatHandler}
        isDeletingChat={isDeletingChat}
      />
      <MainLayoutProvider
        value={{
          hasChat,
          isNewChat,
          isLoadingChats,
          addChatHandler,
          refetchChats,
        }}
      >
        <Outlet />
      </MainLayoutProvider>
    </div>
  );
}
