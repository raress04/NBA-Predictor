export function parseParlays(text) {
  const lines = text.split('\n');
  const blocks = [];
  let currentBlock = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (line.includes('🏀 PARLAY RECOMMENDATION #')) {
      if (currentBlock) blocks.push(currentBlock);
      const titleMatch = line.match(/🏀 PARLAY RECOMMENDATION #(\d+)\s*\((.*?)\)/);
      currentBlock = {
        number: titleMatch ? titleMatch[1] : (blocks.length + 1).toString(),
        picksCount: titleMatch ? titleMatch[2] : '',
        picks: [],
        combinedOdds: null,
        stake: null,
        warning: null,
      };
    } else if (currentBlock && line.startsWith('Pick') && line.includes('[')) {
      const matchupMatch = line.match(/\[(.*?)\]/);
      const isGameTotal = line.includes('] Over') || line.includes('] Under');
      
      let player = 'Game Total';
      let direction = 'Over';
      let lineStat = '';
      
      const parts = line.split(']');
      const rightSide = parts[1] || '';

      if (isGameTotal) {
        const totalMatch = rightSide.match(/(Over|Under)\s+([\d.]+)/);
        direction = totalMatch ? totalMatch[1] : 'Over';
        lineStat = totalMatch ? totalMatch[2] : '';
      } else {
        const propMatch = rightSide.match(/\s*(.*?)\s+(Over|Under)\s+([\d.]+\s+[A-Z\s]+)/);
        player = propMatch ? propMatch[1] : 'Unknown Player';
        direction = propMatch ? propMatch[2] : 'Over';
        lineStat = propMatch ? propMatch[3] : '';
      }

      const simMatch = line.match(/Sim:\s*(\d+)%/);
      const dbMatch = line.match(/→\s*(\d+)%w/);
      const verifiedMatch = line.includes('✅');
      const oddsMatch = line.match(/@\s*([\d.]+)/);

      let edge = null, conf = null, book = null;
      let nextLine = lines[i + 1] ? lines[i + 1].trim() : '';
      if (nextLine.startsWith('Edge:')) {
        const edgeM = nextLine.match(/Edge:\s*([+\-\d.]+%)/);
        const confM = nextLine.match(/Confidence:\s*(\d+%)/);
        const bookM = nextSideMatch(nextLine, 'Book:');
        edge = edgeM ? edgeM[1] : null;
        conf = confM ? confM[1] : null;
        book = bookM;
      }

      currentBlock.picks.push({
        matchup: matchupMatch ? matchupMatch[1] : 'Unknown Matchup',
        player,
        direction,
        lineStat,
        simConf: simMatch ? simMatch[1] : '0',
        dbWinRate: dbMatch ? dbMatch[1] : '0',
        verified: !!verifiedMatch,
        odds: oddsMatch ? oddsMatch[1] : '',
        edge: edge,
        confidenceLine: conf,
        book: book
      });
    } else if (currentBlock && line.includes('Combined Odds')) {
      currentBlock.combinedOdds = line.split(/Combined Odds:?/i)[1]?.trim() || '';
    } else if (currentBlock && line.toLowerCase().includes('recommended stake')) {
      currentBlock.stake = line.split(/Recommended Stake:?/i)[1]?.trim() || '';
    } else if (currentBlock && line.includes('⚠️ [STAKING]')) {
      currentBlock.warning = line.split('⚠️ [STAKING]')[1]?.trim() || '';
    }
  }

  if (currentBlock) {
    blocks.push(currentBlock);
  }

  // Only remove summary/footer blocks, not actual parlays
  return blocks.filter(b => b.picks.length > 0);
}

function nextSideMatch(l, key) {
  if (!l.includes(key)) return null;
  const parts = l.split(key);
  if (parts.length < 2) return null;
  return parts[1].split('|')[0].trim();
}
