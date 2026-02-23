import { create } from 'zustand';
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  type Connection,
  MarkerType,
} from '@xyflow/react';
import type { StageNodeData, StageType, Journey } from '../types/journey';

interface HistoryEntry {
  nodes: Node<StageNodeData>[];
  edges: Edge[];
}

interface JourneyMeta {
  id: string;
  name: string;
  description: string;
  mode: 'pre-sales' | 'post-sales';
  version: number;
  status: 'draft' | 'published';
}

interface JourneyStore {
  nodes: Node<StageNodeData>[];
  edges: Edge[];
  journeyMeta: JourneyMeta;
  selectedNodeId: string | null;
  history: HistoryEntry[];
  historyIndex: number;

  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;

  setSelectedNodeId: (id: string | null) => void;
  addStageNode: (stageType: StageType, position: { x: number; y: number }) => void;
  updateNodeData: (nodeId: string, data: Partial<StageNodeData>) => void;
  deleteNode: (nodeId: string) => void;
  duplicateNode: (nodeId: string) => void;
  deleteEdge: (edgeId: string) => void;
  updateEdgeLabel: (edgeId: string, label: string) => void;

  updateJourneyMeta: (meta: Partial<JourneyMeta>) => void;
  exportJourney: () => Journey;
  importJourney: (journey: Journey) => void;
  clearCanvas: () => void;

  pushHistory: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  autoLayout: () => void;
}

let nodeIdCounter = 0;
const getNextNodeId = () => `stage-${++nodeIdCounter}`;

const defaultMeta: JourneyMeta = {
  id: crypto.randomUUID(),
  name: 'Untitled Journey',
  description: '',
  mode: 'pre-sales',
  version: 1,
  status: 'draft',
};

const stageTypeDefaults: Record<StageType, { label: string; instruction: string }> = {
  start: { label: 'Start', instruction: 'Greet the user and introduce the journey.' },
  stage: { label: 'New Stage', instruction: '' },
  end: { label: 'End', instruction: 'Wrap up the journey and thank the user.' },
};

export const useJourneyStore = create<JourneyStore>((set, get) => ({
  nodes: [],
  edges: [],
  journeyMeta: { ...defaultMeta },
  selectedNodeId: null,
  history: [{ nodes: [], edges: [] }],
  historyIndex: 0,

  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) as Node<StageNodeData>[] });
  },

  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  onConnect: (connection: Connection) => {
    const newEdge: Edge = {
      ...connection,
      id: `edge-${connection.source}-${connection.target}`,
      type: 'smoothstep',
      animated: true,
      markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
      style: { strokeWidth: 2 },
    };
    set({ edges: addEdge(newEdge, get().edges) });
    get().pushHistory();
  },

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  addStageNode: (stageType, position) => {
    const id = getNextNodeId();
    const defaults = stageTypeDefaults[stageType];
    const newNode: Node<StageNodeData> = {
      id,
      type: 'stageNode',
      position,
      data: {
        label: defaults.label,
        instruction: defaults.instruction,
        stageType,
        detectionConfig: { type: 'visual', criteria: '' },
      },
    };
    set({ nodes: [...get().nodes, newNode] });
    get().pushHistory();
  },

  updateNodeData: (nodeId, data) => {
    set({
      nodes: get().nodes.map((node) =>
        node.id === nodeId ? { ...node, data: { ...node.data, ...data } } : node
      ),
    });
    get().pushHistory();
  },

  deleteNode: (nodeId) => {
    set({
      nodes: get().nodes.filter((n) => n.id !== nodeId),
      edges: get().edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
      selectedNodeId: get().selectedNodeId === nodeId ? null : get().selectedNodeId,
    });
    get().pushHistory();
  },

  duplicateNode: (nodeId) => {
    const node = get().nodes.find((n) => n.id === nodeId);
    if (!node) return;
    const id = getNextNodeId();
    const newNode: Node<StageNodeData> = {
      ...node,
      id,
      position: { x: node.position.x + 50, y: node.position.y + 50 },
      data: { ...node.data, label: `${node.data.label} (copy)` },
      selected: false,
    };
    set({ nodes: [...get().nodes, newNode] });
    get().pushHistory();
  },

  deleteEdge: (edgeId) => {
    set({ edges: get().edges.filter((e) => e.id !== edgeId) });
    get().pushHistory();
  },

  updateEdgeLabel: (edgeId, label) => {
    set({
      edges: get().edges.map((e) =>
        e.id === edgeId ? { ...e, label } : e
      ),
    });
    get().pushHistory();
  },

  updateJourneyMeta: (meta) => {
    set({ journeyMeta: { ...get().journeyMeta, ...meta } });
  },

  exportJourney: () => {
    const { nodes, edges, journeyMeta } = get();
    const now = new Date().toISOString();
    return {
      ...journeyMeta,
      stages: nodes.map((n) => ({
        id: n.id,
        name: n.data.label,
        instruction: n.data.instruction,
        detectionConfig: n.data.detectionConfig,
        stageType: n.data.stageType,
        position: n.position,
      })),
      transitions: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        condition: typeof e.label === 'string' ? e.label : undefined,
      })),
      createdAt: now,
      updatedAt: now,
    };
  },

  importJourney: (journey) => {
    const nodes: Node<StageNodeData>[] = journey.stages.map((stage) => ({
      id: stage.id,
      type: 'stageNode',
      position: (stage as unknown as { position: { x: number; y: number } }).position ?? { x: 0, y: 0 },
      data: {
        label: stage.name,
        instruction: stage.instruction,
        stageType: stage.stageType,
        detectionConfig: stage.detectionConfig,
      },
    }));

    const edges: Edge[] = journey.transitions.map((t) => ({
      id: t.id,
      source: t.source,
      target: t.target,
      type: 'smoothstep',
      animated: true,
      label: t.condition,
      markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
      style: { strokeWidth: 2 },
    }));

    const maxId = journey.stages.reduce((max, s) => {
      const match = s.id.match(/stage-(\d+)/);
      return match ? Math.max(max, parseInt(match[1])) : max;
    }, 0);
    nodeIdCounter = maxId;

    set({
      nodes,
      edges,
      journeyMeta: {
        id: journey.id,
        name: journey.name,
        description: journey.description,
        mode: journey.mode,
        version: journey.version,
        status: journey.status,
      },
      selectedNodeId: null,
      history: [{ nodes, edges }],
      historyIndex: 0,
    });
  },

  clearCanvas: () => {
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      journeyMeta: { ...defaultMeta, id: crypto.randomUUID() },
    });
    get().pushHistory();
  },

  pushHistory: () => {
    const { nodes, edges, history, historyIndex } = get();
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    if (newHistory.length > 50) newHistory.shift();
    set({ history: newHistory, historyIndex: newHistory.length - 1 });
  },

  undo: () => {
    const { historyIndex, history } = get();
    if (historyIndex <= 0) return;
    const newIndex = historyIndex - 1;
    const entry = history[newIndex];
    set({
      nodes: JSON.parse(JSON.stringify(entry.nodes)),
      edges: JSON.parse(JSON.stringify(entry.edges)),
      historyIndex: newIndex,
    });
  },

  redo: () => {
    const { historyIndex, history } = get();
    if (historyIndex >= history.length - 1) return;
    const newIndex = historyIndex + 1;
    const entry = history[newIndex];
    set({
      nodes: JSON.parse(JSON.stringify(entry.nodes)),
      edges: JSON.parse(JSON.stringify(entry.edges)),
      historyIndex: newIndex,
    });
  },

  canUndo: () => get().historyIndex > 0,
  canRedo: () => get().historyIndex < get().history.length - 1,

  autoLayout: () => {
    const { nodes, edges } = get();
    if (nodes.length === 0) return;

    // Find root nodes (no incoming edges)
    const targets = new Set(edges.map((e) => e.target));
    const roots = nodes.filter((n) => !targets.has(n.id));
    if (roots.length === 0 && nodes.length > 0) {
      roots.push(nodes[0]);
    }

    // BFS to assign levels
    const levels: Map<string, number> = new Map();
    const queue: string[] = [];
    const adjacency: Map<string, string[]> = new Map();

    for (const edge of edges) {
      if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
      adjacency.get(edge.source)!.push(edge.target);
    }

    for (const root of roots) {
      levels.set(root.id, 0);
      queue.push(root.id);
    }

    while (queue.length > 0) {
      const current = queue.shift()!;
      const level = levels.get(current) ?? 0;
      const children = adjacency.get(current) ?? [];
      for (const child of children) {
        if (!levels.has(child)) {
          levels.set(child, level + 1);
          queue.push(child);
        }
      }
    }

    // Assign positions to unvisited nodes
    let maxLevel = 0;
    for (const node of nodes) {
      if (!levels.has(node.id)) {
        levels.set(node.id, ++maxLevel);
      }
      maxLevel = Math.max(maxLevel, levels.get(node.id)!);
    }

    // Group by level
    const byLevel: Map<number, string[]> = new Map();
    for (const [id, level] of levels) {
      if (!byLevel.has(level)) byLevel.set(level, []);
      byLevel.get(level)!.push(id);
    }

    const HORIZONTAL_GAP = 300;
    const VERTICAL_GAP = 150;

    const layoutedNodes = nodes.map((node) => {
      const level = levels.get(node.id) ?? 0;
      const siblings = byLevel.get(level) ?? [node.id];
      const index = siblings.indexOf(node.id);
      const totalWidth = (siblings.length - 1) * VERTICAL_GAP;

      return {
        ...node,
        position: {
          x: level * HORIZONTAL_GAP + 50,
          y: index * VERTICAL_GAP - totalWidth / 2 + 300,
        },
      };
    });

    set({ nodes: layoutedNodes });
    get().pushHistory();
  },
}));
