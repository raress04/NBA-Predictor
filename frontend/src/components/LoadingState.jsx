export default function LoadingState() {
  return (
    <div className="space-y-6 animate-pulse w-full max-w-[760px] mx-auto">
      <div className="bg-[#111118] border border-[#2A2A3A] rounded-xl p-6 shadow-glow">
        <div className="flex justify-between items-center mb-6">
          <div className="h-6 w-32 bg-[#1A1A24] rounded-full"></div>
          <div className="h-6 w-20 bg-accent-primary/20 rounded-full"></div>
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex space-x-4">
              <div className="w-1/3 h-6 bg-[#1A1A24] rounded"></div>
              <div className="w-1/2 h-6 bg-[#1A1A24] rounded"></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
