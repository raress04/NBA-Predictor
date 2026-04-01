import { useState, useEffect } from 'react';
import { getTodayFilePath } from '../utils/dateHelpers';
import { parseParlays } from '../utils/parseParlay';
import LoadingState from './LoadingState';
import ErrorState from './ErrorState';
import ParlayCard from './ParlayCard';

export default function PredictionContainer() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [parlays, setParlays] = useState([]);

  useEffect(() => {
    async function fetchPredictions() {
      try {
        const filePath = getTodayFilePath();
        console.log('Fetching prediction from:', filePath);
        const response = await fetch(filePath);
        console.log('Response status:', response.status);
        
        if (!response.ok) {
          if (response.status === 404) {
            console.warn('Prediction file not found at:', filePath);
            setError(true);
          } else {
            throw new Error(`Network Error: ${response.status}`);
          }
          setLoading(false);
          return;
        }
        
        const text = await response.text();
        console.log('Fetched text length:', text.length);
        const parsed = parseParlays(text);
        console.log('Parsed parlays:', parsed);
        
        if (parsed.length === 0) {
          console.warn('Parser returned 0 results.');
          setError(true);
        } else {
          setParlays(parsed);
        }
      } catch (err) {
        console.error('Error during fetch/parse:', err);
        setError(true);
      } finally {
        setLoading(false);
      }
    }

    fetchPredictions();
  }, []);

  if (loading) {
    return (
      <div className="w-full flex justify-center py-10 px-4">
        <LoadingState />
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full flex justify-center py-10 px-4">
        <ErrorState />
      </div>
    );
  }

  return (
    <div className="w-full flex flex-col items-center py-10 px-4">
      <div className="w-full max-w-[760px]">
        {parlays.map((block, index) => (
          <ParlayCard key={index} index={index} block={block} />
        ))}
      </div>
    </div>
  );
}
