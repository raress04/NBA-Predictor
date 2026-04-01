export function getTodayFilePath() {
  const today = new Date();
  const month = today.toLocaleString('en-US', { month: 'long' }).toLowerCase();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, '0');
  const dd = String(today.getDate()).padStart(2, '0');
  
  return `/predictions/${month}/predicts_${yyyy}-${mm}-${dd}.txt`;
}

export function getFormattedDate() {
  const today = new Date();
  return today.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric'
  });
}
