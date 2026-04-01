import { Sparkles } from 'lucide-react';

export default function ParlayHeader({ number, picksCount, isFeatured }) {
  return (
    <div className="flex justify-between items-center mb-6">
      <div className="flex items-center space-x-3">
        <h3 className="text-white font-bold text-lg tracking-wide uppercase">
          Parlay #{number}
        </h3>
        <span className="px-2.5 py-1 bg-[#1A1A24] text-text-muted rounded border border-[#2A2A3A] text-[11px] font-bold uppercase tracking-wider">
          {picksCount} Picks
        </span>
      </div>
      {isFeatured && (
        <div className="flex items-center space-x-1.5 px-3 py-1 bg-accent-primary/10 border border-accent-primary/20 rounded-full text-accent-primary text-xs font-bold uppercase tracking-wider">
          <Sparkles size={12} className="text-accent-primary" />
          <span>Featured</span>
        </div>
      )}
    </div>
  );
}
