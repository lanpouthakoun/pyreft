import { useCallback, useRef, useEffect, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type ReactFlowInstance,
  type Node,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './App.css';

import StageNode from './components/StageNode';
import Sidebar from './components/Sidebar';
import PropertyPanel from './components/PropertyPanel';
import Toolbar from './components/Toolbar';
import { useJourneyStore } from './store/journeyStore';
import type { StageType, StageNodeData } from './types/journey';

function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addStageNode,
    setSelectedNodeId,
    undo,
    redo,
  } = useJourneyStore();

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const reactFlowInstance = useRef<ReactFlowInstance<Node<StageNodeData>> | null>(null);

  const nodeTypes = useMemo(() => ({ stageNode: StageNode }), []);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const stageType = event.dataTransfer.getData('application/stageType') as StageType;
      if (!stageType) return;

      const bounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!bounds || !reactFlowInstance.current) return;

      const position = reactFlowInstance.current.screenToFlowPosition({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });

      addStageNode(stageType, position);
    },
    [addStageNode]
  );

  const onInit = useCallback((instance: ReactFlowInstance<Node<StageNodeData>>) => {
    reactFlowInstance.current = instance;
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, [setSelectedNodeId]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo]);

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-50">
      <Toolbar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <div className="flex-1 relative" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={onInit}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            fitView
            snapToGrid
            snapGrid={[16, 16]}
            deleteKeyCode={['Backspace', 'Delete']}
            className="bg-slate-50"
          >
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#cbd5e1" />
            <Controls className="!bg-white !border-slate-200 !shadow-md !rounded-lg" />
            <MiniMap
              nodeColor={(node) => {
                const stageType = (node.data as { stageType?: string })?.stageType;
                switch (stageType) {
                  case 'start': return '#10b981';
                  case 'end': return '#f43f5e';
                  default: return '#3b82f6';
                }
              }}
              className="!bg-white !border-slate-200 !shadow-md !rounded-lg"
              maskColor="rgba(0, 0, 0, 0.08)"
            />
          </ReactFlow>
        </div>
        <PropertyPanel />
      </div>
    </div>
  );
}

export default App;
