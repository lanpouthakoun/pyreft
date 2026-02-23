import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Play, Square, Circle, Trash2, Copy } from 'lucide-react';
import type { StageNodeData, StageType } from '../types/journey';
import { useJourneyStore } from '../store/journeyStore';

const stageColors: Record<StageType, { bg: string; border: string; icon: string }> = {
  start: { bg: 'bg-emerald-50', border: 'border-emerald-400', icon: 'text-emerald-600' },
  stage: { bg: 'bg-blue-50', border: 'border-blue-400', icon: 'text-blue-600' },
  end: { bg: 'bg-rose-50', border: 'border-rose-400', icon: 'text-rose-600' },
};

const StageIcon = ({ stageType }: { stageType: StageType }) => {
  const size = 16;
  switch (stageType) {
    case 'start':
      return <Play size={size} />;
    case 'end':
      return <Square size={size} />;
    default:
      return <Circle size={size} />;
  }
};

function StageNode({ id, data, selected }: NodeProps) {
  const nodeData = data as StageNodeData;
  const { deleteNode, duplicateNode, setSelectedNodeId } = useJourneyStore();
  const colors = stageColors[nodeData.stageType];

  return (
    <div
      className={`
        relative rounded-lg border-2 shadow-md min-w-56 max-w-72
        ${colors.bg} ${colors.border}
        ${selected ? 'ring-2 ring-indigo-500 ring-offset-2' : ''}
        transition-shadow duration-150
      `}
      onClick={() => setSelectedNodeId(id)}
    >
      {nodeData.stageType !== 'start' && (
        <Handle
          type="target"
          position={Position.Left}
          className="!w-3 !h-3 !bg-slate-400 !border-2 !border-white"
        />
      )}

      <div className={`flex items-center gap-2 px-3 py-2 border-b ${colors.border} bg-white/50 rounded-t-lg`}>
        <span className={colors.icon}>
          <StageIcon stageType={nodeData.stageType} />
        </span>
        <span className="font-semibold text-sm text-slate-800 truncate flex-1">
          {nodeData.label}
        </span>
        <span className="text-xs text-slate-400 uppercase tracking-wider">
          {nodeData.stageType}
        </span>
      </div>

      <div className="px-3 py-2">
        {nodeData.instruction ? (
          <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
            {nodeData.instruction}
          </p>
        ) : (
          <p className="text-xs text-slate-400 italic">No instruction set</p>
        )}

        {nodeData.detectionConfig.criteria && (
          <div className="mt-1.5 flex items-center gap-1">
            <span className="text-xs px-1.5 py-0.5 rounded bg-slate-200 text-slate-600">
              {nodeData.detectionConfig.type}
            </span>
          </div>
        )}
      </div>

      {selected && (
        <div className="absolute -top-9 right-0 flex gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); duplicateNode(id); }}
            className="p-1 rounded bg-white shadow border border-slate-200 hover:bg-slate-50 text-slate-500"
            title="Duplicate"
          >
            <Copy size={14} />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); deleteNode(id); }}
            className="p-1 rounded bg-white shadow border border-red-200 hover:bg-red-50 text-red-500"
            title="Delete"
          >
            <Trash2 size={14} />
          </button>
        </div>
      )}

      {nodeData.stageType !== 'end' && (
        <Handle
          type="source"
          position={Position.Right}
          className="!w-3 !h-3 !bg-slate-400 !border-2 !border-white"
        />
      )}
    </div>
  );
}

export default memo(StageNode);
