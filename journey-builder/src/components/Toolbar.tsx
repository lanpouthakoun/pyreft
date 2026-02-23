import { Undo2, Redo2, Save, Edit3 } from 'lucide-react';
import { useJourneyStore } from '../store/journeyStore';
import { useState } from 'react';

export default function Toolbar() {
  const { undo, redo, canUndo, canRedo, journeyMeta, updateJourneyMeta, exportJourney } = useJourneyStore();
  const [editingName, setEditingName] = useState(false);
  const [nameValue, setNameValue] = useState(journeyMeta.name);

  const handleSave = () => {
    const journey = exportJourney();
    localStorage.setItem(`journey-${journey.id}`, JSON.stringify(journey));
    const saved = document.getElementById('save-indicator');
    if (saved) {
      saved.classList.remove('opacity-0');
      saved.classList.add('opacity-100');
      setTimeout(() => {
        saved.classList.remove('opacity-100');
        saved.classList.add('opacity-0');
      }, 2000);
    }
  };

  const commitName = () => {
    updateJourneyMeta({ name: nameValue });
    setEditingName(false);
  };

  return (
    <div className="h-12 bg-white border-b border-slate-200 flex items-center justify-between px-4 shrink-0">
      <div className="flex items-center gap-3">
        {editingName ? (
          <input
            autoFocus
            value={nameValue}
            onChange={(e) => setNameValue(e.target.value)}
            onBlur={commitName}
            onKeyDown={(e) => { if (e.key === 'Enter') commitName(); if (e.key === 'Escape') { setNameValue(journeyMeta.name); setEditingName(false); } }}
            className="text-sm font-semibold text-slate-800 border border-indigo-300 rounded px-2 py-0.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        ) : (
          <button
            onClick={() => { setNameValue(journeyMeta.name); setEditingName(true); }}
            className="flex items-center gap-1.5 text-sm font-semibold text-slate-800 hover:text-indigo-600 transition-colors"
          >
            {journeyMeta.name}
            <Edit3 size={12} className="text-slate-400" />
          </button>
        )}

        <select
          value={journeyMeta.mode}
          onChange={(e) => updateJourneyMeta({ mode: e.target.value as 'pre-sales' | 'post-sales' })}
          className="text-xs px-2 py-1 rounded border border-slate-200 text-slate-600 bg-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
        >
          <option value="pre-sales">Pre-Sales</option>
          <option value="post-sales">Post-Sales</option>
        </select>

        <span className={`text-xs px-2 py-0.5 rounded-full ${
          journeyMeta.status === 'draft'
            ? 'bg-amber-100 text-amber-700'
            : 'bg-green-100 text-green-700'
        }`}>
          {journeyMeta.status}
        </span>
      </div>

      <div className="flex items-center gap-1">
        <span id="save-indicator" className="text-xs text-green-600 mr-2 opacity-0 transition-opacity duration-300">
          Saved
        </span>

        <button
          onClick={undo}
          disabled={!canUndo()}
          className="p-2 rounded hover:bg-slate-100 text-slate-600 disabled:text-slate-300 disabled:hover:bg-transparent transition-colors"
          title="Undo (Ctrl+Z)"
        >
          <Undo2 size={16} />
        </button>
        <button
          onClick={redo}
          disabled={!canRedo()}
          className="p-2 rounded hover:bg-slate-100 text-slate-600 disabled:text-slate-300 disabled:hover:bg-transparent transition-colors"
          title="Redo (Ctrl+Shift+Z)"
        >
          <Redo2 size={16} />
        </button>

        <div className="w-px h-6 bg-slate-200 mx-1" />

        <button
          onClick={handleSave}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors"
        >
          <Save size={14} />
          Save
        </button>
      </div>
    </div>
  );
}
