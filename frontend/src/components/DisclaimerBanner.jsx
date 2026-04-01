import { useState, useEffect } from 'react';
import { X } from 'lucide-react';

export default function DisclaimerBanner() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const dismissed = sessionStorage.getItem('disclaimerDismissed');
    if (!dismissed) {
      setIsVisible(true);
    }
  }, []);

  if (!isVisible) return null;

  return (
    <div role="alert" className="w-full h-10 bg-[#1A0A0A] border-l-4 border-[#7F1D1D] flex items-center justify-between px-4 sm:px-6 relative shadow-sm z-10 transition-all">
      <div className="flex-1 text-center text-[#9B9B9B] text-xs font-medium tracking-wide">
        For entertainment and analytical purposes only. 18+. ONJN licensed operators only. Please bet responsibly.
      </div>
      <button 
        onClick={() => {
          sessionStorage.setItem('disclaimerDismissed', 'true');
          setIsVisible(false);
        }}
        aria-label="Dismiss disclaimer"
        className="text-[#9B9B9B] hover:text-white transition-colors"
      >
        <X size={14} />
      </button>
    </div>
  );
}
