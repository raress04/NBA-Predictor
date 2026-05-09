SELECT COUNT(*), MIN(parlay_date), MAX(parlay_date), COUNT(CASE WHEN result='PENDING' THEN 1 END) as pending FROM parlays;
