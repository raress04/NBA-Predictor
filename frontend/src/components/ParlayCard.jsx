import { motion } from 'framer-motion';
import ParlayHeader from './ParlayHeader';
import PickRow from './PickRow';
import ParlayFooter from './ParlayFooter';

export default function ParlayCard({ block, index }) {
  const isFeatured = index === 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut", staggerChildren: 0.1 }}
      className={`relative w-full mb-8 p-6 bg-[#111118] border border-[#2A2A3A] rounded-xl overflow-hidden group hover:shadow-glow-hover hover:-translate-y-1 transition-all duration-300 ${isFeatured ? 'shadow-glow border-accent-primary/20' : 'shadow-none'}`}
    >
      <ParlayHeader 
        number={block.number} 
        picksCount={block.picksCount} 
        isFeatured={isFeatured} 
      />

      <div className="space-y-4">
        {block.picks.map((pick, i) => (
          <PickRow key={i} pick={pick} />
        ))}
      </div>

      <ParlayFooter 
        combinedOdds={block.combinedOdds}
        stake={block.stake}
        warning={block.warning}
      />
    </motion.div>
  );
}
