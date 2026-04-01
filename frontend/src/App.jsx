import Navbar from './components/Navbar';
import DisclaimerBanner from './components/DisclaimerBanner';
import PredictionContainer from './components/PredictionContainer';

function App() {
  return (
    <div className="flex flex-col min-h-screen bg-background relative w-full overflow-x-hidden pt-10">
      <DisclaimerBanner />
      <Navbar />

      <main className="flex-grow flex flex-col justify-start pb-20 mt-4 px-2 w-full">
        <PredictionContainer />
      </main>

      <footer className="h-16 w-full flex items-center justify-center border-t border-[#1A1A24] bg-[#0A0A0FCC]">
        <p className="text-text-muted text-xs tracking-wider font-mono">
          © {new Date().getFullYear()} AI Parlay. For entertainment purposes only.
        </p>
      </footer>
    </div>
  );
}

export default App;
