import React, { useState } from 'react';
import { usePWAInstall } from '../hooks/usePWAInstall';

export const PWAInstallButton: React.FC = () => {
  const { isInstallable, isInstalled, isIOS, install } = usePWAInstall();
  const [showIOSGuide, setShowIOSGuide] = useState(false);

  if (isInstalled) {
    return null;
  }

  if (isInstallable) {
    return (
      <button
        id="btnInstallApp"
        type="button"
        onClick={install}
        title="Install RL Evasion Baseline as a standalone app"
        className="bg-[#222] text-[#0ff] border border-[#066] px-[8px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer hover:bg-[#066] hover:text-[#eee] whitespace-nowrap"
      >
        Install App
      </button>
    );
  }

  if (isIOS) {
    return (
      <>
        <button
          id="btnInstallIOS"
          type="button"
          onClick={() => setShowIOSGuide(true)}
          title="Install RL Evasion Baseline on iOS"
          className="bg-[#222] text-[#0ff] border border-[#066] px-[8px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer hover:bg-[#066] hover:text-[#eee] whitespace-nowrap"
        >
          Install App
        </button>

        {showIOSGuide && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 font-mono">
            <div className="w-full max-w-[320px] bg-[#161616] border border-[#444] p-[12px] text-[#eee] text-[12px]">
              <div className="text-[#0ff] font-bold mb-[6px]">Install on iPhone / iPad</div>
              <p className="text-[#ccc] leading-relaxed mb-[10px]">
                1. Tap the <strong>Share</strong> button in Safari&apos;s toolbar.
                <br />
                2. Scroll down and tap <strong>Add to Home Screen</strong>.
              </p>
              <button
                type="button"
                onClick={() => setShowIOSGuide(false)}
                className="w-full bg-[#222] text-[#eee] border border-[#444] p-[4px] font-mono text-[12px] cursor-pointer hover:bg-[#333]"
              >
                Close
              </button>
            </div>
          </div>
        )}
      </>
    );
  }

  return null;
};
