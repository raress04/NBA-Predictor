import { getFormattedDate } from '../utils/dateHelpers';

export default function Navbar() {
  return (
    <nav className="h-14 bg-[#0A0A0F] border-b border-[#1A1A24] flex items-center justify-between px-4 sm:px-6 w-full">
      <div className="flex items-center space-x-3">
        <span className="text-xl">🏀</span>
        <span className="font-bold text-white tracking-tight">AI Parlay</span>
        <span className="px-2 py-0.5 rounded-full bg-accent-primary/20 text-accent-primary text-[10px] font-bold tracking-wider uppercase">Beta</span>
      </div>
      <div className="text-text-muted text-sm font-medium">
        {getFormattedDate()} <span className="hidden sm:inline">· ERA3</span>
      </div>
    </nav>
  );
}
