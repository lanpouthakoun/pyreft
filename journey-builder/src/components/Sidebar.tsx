import { type DragEvent } from 'react';
import { Play, Circle, Square, FileJson, FolderOpen, Trash2, LayoutGrid } from 'lucide-react';
import type { StageType } from '../types/journey';
import { useJourneyStore } from '../store/journeyStore';

const stageItems: { type: StageType; label: string; description: string; icon: typeof Play; color: string }[] = [
  { type: 'start', label: 'Start Stage', description: 'Entry point of journey', icon: Play, color: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  { type: 'stage', label: 'Stage', description: 'Journey step', icon: Circle, color: 'text-blue-600 bg-blue-50 border-blue-200' },
  { type: 'end', label: 'End Stage', description: 'Journey conclusion', icon: Square, color: 'text-rose-600 bg-rose-50 border-rose-200' },
];

function DraggableStage({ type, label, description, icon: Icon, color }: typeof stageItems[number]) {
  const onDragStart = (e: DragEvent) => {
    e.dataTransfer.setData('application/stageType', type);
    e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className={`flex items-center gap-3 p-3 rounded-lg border cursor-grab active:cursor-grabbing hover:shadow-sm transition-shadow ${color}`}
    >
      <Icon size={20} />
      <div>
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs opacity-60">{description}</p>
      </div>
    </div>
  );
}

export default function Sidebar() {
  const { exportJourney, importJourney, clearCanvas, autoLayout } = useJourneyStore();

  const handleExport = () => {
    const journey = exportJourney();
    const blob = new Blob([JSON.stringify(journey, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${journey.name.toLowerCase().replace(/\s+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const journey = JSON.parse(ev.target?.result as string);
          importJourney(journey);
        } catch {
          alert('Invalid journey JSON file');
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col h-full">
      <div className="p-4 border-b border-slate-200">
        <h2 className="text-lg font-bold text-slate-800">Journey Builder</h2>
        <p className="text-xs text-slate-500 mt-0.5">Drag stages onto the canvas</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Stage Types</h3>
        {stageItems.map((item) => (
          <DraggableStage key={item.type} {...item} />
        ))}
      </div>

      <div className="p-4 border-t border-slate-200 space-y-2">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Actions</h3>
        <button
          onClick={autoLayout}
          className="flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 transition-colors"
        >
          <LayoutGrid size={16} />
          Auto Layout
        </button>
        <button
          onClick={handleExport}
          className="flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 transition-colors"
        >
          <FileJson size={16} />
          Export JSON
        </button>
        <button
          onClick={handleImport}
          className="flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 transition-colors"
        >
          <FolderOpen size={16} />
          Import JSON
        </button>
        <button
          onClick={() => {
            if (confirm('Clear entire canvas? This cannot be undone.')) clearCanvas();
          }}
          className="flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg border border-red-200 hover:bg-red-50 text-red-600 transition-colors"
        >
          <Trash2 size={16} />
          Clear Canvas
        </button>
      </div>
    </aside>
  );
}
