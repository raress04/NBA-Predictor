import { TriangleAlert } from 'lucide-react';

export default function ParlayFooter({ combinedOdds, stake, warning }) {
  if (!combinedOdds && !stake && !warning) return null;

  return (
    <div className="mt-8 border-t border-dashed border-[#2A2A3A] pt-5">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center w-full gap-4">
        {combinedOdds && (
          <div className="flex flex-col">
            <span className="text-text-muted text-xs font-bold uppercase tracking-widest mb-1">
              Combined Odds
            </span>
            <span className="font-mono text-xl sm:text-2xl font-bold text-white tracking-widest drop-shadow-md">
              {combinedOdds}
            </span>
          </div>
        )}

        {stake && (
          <div className="flex flex-col sm:items-end">
            <span className="text-text-muted text-xs font-bold uppercase tracking-widest mb-1">
              Recommended Stake
            </span>
            <span className="font-bold text-accent-primary text-base sm:text-lg bg-accent-primary/10 px-3 py-1 rounded-sm tracking-wide">
              {stake}
            </span>
          </div>
        )}
      </div>

      {warning && (
        <div className="mt-5 bg-[#1A1A24]/60 border border-accent-amber/20 px-4 py-3 rounded-md flex items-start space-x-3 text-xs italic text-accent-amber">
          <TriangleAlert size={16} className="mt-0.5 flex-shrink-0" />
          <span className="font-medium tracking-wide">{warning.replace('⚠️ ', '')}</span>
        </div>
      )}
    </div>
  );
}
