import { useJourneyStore } from '../store/journeyStore';
import type { StageNodeData, DetectionType, StageType } from '../types/journey';
import { X, Settings } from 'lucide-react';

const detectionTypes: { value: DetectionType; label: string }[] = [
  { value: 'visual', label: 'Visual' },
  { value: 'text', label: 'Text' },
  { value: 'url', label: 'URL' },
  { value: 'custom', label: 'Custom' },
];

const stageTypes: { value: StageType; label: string }[] = [
  { value: 'start', label: 'Start' },
  { value: 'stage', label: 'Stage' },
  { value: 'end', label: 'End' },
];

export default function PropertyPanel() {
  const { nodes, selectedNodeId, setSelectedNodeId, updateNodeData } = useJourneyStore();
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  if (!selectedNode) {
    return (
      <aside className="w-80 bg-white border-l border-slate-200 flex flex-col h-full">
        <div className="flex-1 flex items-center justify-center text-slate-400">
          <div className="text-center space-y-2">
            <Settings size={32} className="mx-auto opacity-40" />
            <p className="text-sm">Select a stage to edit its properties</p>
          </div>
        </div>
      </aside>
    );
  }

  const data = selectedNode.data as StageNodeData;

  const handleChange = (field: string, value: string) => {
    if (field === 'detectionType') {
      updateNodeData(selectedNode.id, {
        detectionConfig: { ...data.detectionConfig, type: value as DetectionType },
      });
    } else if (field === 'detectionCriteria') {
      updateNodeData(selectedNode.id, {
        detectionConfig: { ...data.detectionConfig, criteria: value },
      });
    } else if (field === 'stageType') {
      updateNodeData(selectedNode.id, { stageType: value as StageType });
    } else {
      updateNodeData(selectedNode.id, { [field]: value });
    }
  };

  return (
    <aside className="w-80 bg-white border-l border-slate-200 flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b border-slate-200">
        <h3 className="font-semibold text-slate-800">Stage Properties</h3>
        <button
          onClick={() => setSelectedNodeId(null)}
          className="p-1 rounded hover:bg-slate-100 text-slate-400"
        >
          <X size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        <div>
          <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
            Stage Name
          </label>
          <input
            type="text"
            value={data.label}
            onChange={(e) => handleChange('label', e.target.value)}
            className="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            placeholder="Stage name"
          />
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
            Stage Type
          </label>
          <select
            value={data.stageType}
            onChange={(e) => handleChange('stageType', e.target.value)}
            className="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
          >
            {stageTypes.map((st) => (
              <option key={st.value} value={st.value}>{st.label}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
            Instruction
          </label>
          <textarea
            value={data.instruction}
            onChange={(e) => handleChange('instruction', e.target.value)}
            rows={5}
            className="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-y"
            placeholder="Instructions for the agent at this stage..."
          />
        </div>

        <div className="border-t border-slate-200 pt-4">
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Detection Config
          </h4>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">
                Detection Type
              </label>
              <select
                value={data.detectionConfig.type}
                onChange={(e) => handleChange('detectionType', e.target.value)}
                className="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
              >
                {detectionTypes.map((dt) => (
                  <option key={dt.value} value={dt.value}>{dt.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">
                Detection Criteria
              </label>
              <textarea
                value={data.detectionConfig.criteria}
                onChange={(e) => handleChange('detectionCriteria', e.target.value)}
                rows={3}
                className="w-full px-3 py-2 text-sm rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-y"
                placeholder={
                  data.detectionConfig.type === 'url'
                    ? 'URL pattern to match...'
                    : data.detectionConfig.type === 'visual'
                    ? 'Visual elements to detect...'
                    : data.detectionConfig.type === 'text'
                    ? 'Text content to match...'
                    : 'Custom detection logic...'
                }
              />
            </div>
          </div>
        </div>

        <div className="border-t border-slate-200 pt-4">
          <p className="text-xs text-slate-400">
            Node ID: <code className="bg-slate-100 px-1 py-0.5 rounded">{selectedNode.id}</code>
          </p>
        </div>
      </div>
    </aside>
  );
}
