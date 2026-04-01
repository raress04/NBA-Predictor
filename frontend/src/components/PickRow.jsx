import { motion } from 'framer-motion';
import { CheckCircle2 } from 'lucide-react';

export default function PickRow({ pick }) {
  const isOver = pick.direction?.toLowerCase() === 'over';
  const confidenceValue = parseInt(pick.simConf) || 0;
  
  return (
    <motion.div 
      className="mb-5 last:mb-0 relative group"
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
    >
      <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-3 mb-2">
        <div className="flex-shrink-0 bg-[#1A1A24] border border-[#2A2A3A] px-2 py-1 rounded-md text-[10px] font-mono text-text-muted tracking-tight w-fit">
          {pick.matchup}
        </div>
        <div className="flex-1 flex flex-wrap items-center gap-x-2 text-[15px] font-medium text-text-primary">
          <span>{pick.player}</span>
          <span className={`font-bold ${isOver ? 'text-accent-green' : 'text-accent-amber'}`}>
            {pick.direction} {pick.lineStat}
          </span>
          <span className="font-mono text-sm pl-1 opacity-80">@{pick.odds}</span>
          {pick.verified && (
            <div className="flex items-center space-x-1 pl-1" title="Verified Edge">
              <CheckCircle2 size={14} className="text-accent-green drop-shadow-md" />
              <span className="text-[10px] uppercase font-bold text-accent-green opacity-90 tracking-wider">Verified</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1.5 px-1">
        {pick.edge && (
          <span className="text-xs text-text-secondary font-mono bg-accent-primary/5 px-2 py-0.5 rounded text-accent-primary/90">
            Edge: <span className="font-bold">{pick.edge}</span>
          </span>
        )}
        {pick.confidenceLine && (
          <>
            <span className="text-text-muted text-xs">|</span>
            <span className="text-xs text-text-secondary">
              Confidence: {pick.confidenceLine}
            </span>
          </>
        )}
        {pick.book && (
          <>
            <span className="text-text-muted text-xs hidden sm:inline">|</span>
            <span className="text-[10px] uppercase tracking-wider text-text-muted font-bold ml-auto sm:ml-0 bg-[#1A1A24] px-1.5 py-0.5 rounded">
              {pick.book}
            </span>
          </>
        )}
      </div>

      <div className="w-full h-1 bg-[#1A1A24] rounded-full overflow-hidden mt-3 relative mb-1">
        <motion.div 
          className={`h-full rounded-full ${isOver ? 'bg-accent-green' : 'bg-accent-amber'}`}
          initial={{ width: "0%" }}
          animate={{ width: `${confidenceValue}%` }}
          transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
        />
      </div>
    </motion.div>
  );
}
