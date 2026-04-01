import { Clock } from 'lucide-react';
import { getFormattedDate } from '../utils/dateHelpers';

export default function ErrorState() {
  return (
    <div className="flex flex-col items-center justify-center p-12 mt-12 w-full max-w-[680px] mx-auto bg-[#111118] border border-[#2A2A3A] rounded-xl shadow-lg">
      <div className="bg-[#1A1A24] rounded-full p-4 mb-4 text-text-muted">
        <Clock size={32} />
      </div>
      <h2 className="text-text-primary text-lg font-semibold mb-2">No prediction available yet</h2>
      <p className="text-text-secondary text-sm text-center mb-6">
        Today's parlay is generated daily before market open. Check back soon.
      </p>
      <div className="px-4 py-2 bg-[#1A1A24] rounded border border-[#2A2A3A] text-text-muted font-mono text-xs">
        Checking date: {getFormattedDate()}
      </div>
    </div>
  );
}
