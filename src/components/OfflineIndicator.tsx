import React from 'react';
import { useOnlineStatus } from '../hooks/useOnlineStatus';

export const OfflineIndicator: React.FC = () => {
  const isOnline = useOnlineStatus();

  if (isOnline) return null;

  return (
    <div className="fixed bottom-3 left-3 z-50 flex items-center gap-2 bg-[#161616] border border-[#f59e0b] px-[8px] py-[4px] font-mono text-[11px] text-[#fbbf24]">
      <span className="h-[6px] w-[6px] bg-[#fbbf24] inline-block" />
      Offline Mode
    </div>
  );
};
