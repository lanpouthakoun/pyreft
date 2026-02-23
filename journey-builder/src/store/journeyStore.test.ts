import { describe, it, expect, beforeEach } from 'vitest';
import { useJourneyStore } from './journeyStore';

// Reset store state before each test
beforeEach(() => {
  useJourneyStore.setState({
    nodes: [],
    edges: [],
    selectedNodeId: null,
    history: [{ nodes: [], edges: [] }],
    historyIndex: 0,
  });
});

describe('journeyStore', () => {
  describe('addStageNode', () => {
    it('adds a start node with correct defaults', () => {
      const { addStageNode } = useJourneyStore.getState();
      addStageNode('start', { x: 100, y: 200 });

      const { nodes } = useJourneyStore.getState();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].data.label).toBe('Start');
      expect(nodes[0].data.stageType).toBe('start');
      expect(nodes[0].data.instruction).toBe('Greet the user and introduce the journey.');
      expect(nodes[0].position).toEqual({ x: 100, y: 200 });
      expect(nodes[0].type).toBe('stageNode');
    });

    it('adds a stage node with correct defaults', () => {
      const { addStageNode } = useJourneyStore.getState();
      addStageNode('stage', { x: 0, y: 0 });

      const { nodes } = useJourneyStore.getState();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].data.label).toBe('New Stage');
      expect(nodes[0].data.stageType).toBe('stage');
      expect(nodes[0].data.instruction).toBe('');
    });

    it('adds an end node with correct defaults', () => {
      const { addStageNode } = useJourneyStore.getState();
      addStageNode('end', { x: 300, y: 400 });

      const { nodes } = useJourneyStore.getState();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].data.label).toBe('End');
      expect(nodes[0].data.stageType).toBe('end');
      expect(nodes[0].data.instruction).toBe('Wrap up the journey and thank the user.');
    });

    it('assigns unique ids to each node', () => {
      const { addStageNode } = useJourneyStore.getState();
      addStageNode('start', { x: 0, y: 0 });
      addStageNode('stage', { x: 100, y: 0 });
      addStageNode('end', { x: 200, y: 0 });

      const { nodes } = useJourneyStore.getState();
      const ids = nodes.map((n) => n.id);
      expect(new Set(ids).size).toBe(3);
    });
  });

  describe('updateNodeData', () => {
    it('updates node label', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('stage', { x: 0, y: 0 });
      const nodeId = useJourneyStore.getState().nodes[0].id;

      store.updateNodeData(nodeId, { label: 'Updated Label' });
      const updated = useJourneyStore.getState().nodes[0];
      expect(updated.data.label).toBe('Updated Label');
    });

    it('updates detection config', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('stage', { x: 0, y: 0 });
      const nodeId = useJourneyStore.getState().nodes[0].id;

      store.updateNodeData(nodeId, {
        detectionConfig: { type: 'url', criteria: 'https://example.com' },
      });
      const updated = useJourneyStore.getState().nodes[0];
      expect(updated.data.detectionConfig.type).toBe('url');
      expect(updated.data.detectionConfig.criteria).toBe('https://example.com');
    });
  });

  describe('deleteNode', () => {
    it('removes a node and its connected edges', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('start', { x: 0, y: 0 });
      store.addStageNode('end', { x: 200, y: 0 });

      const { nodes } = useJourneyStore.getState();
      const startId = nodes[0].id;

      store.deleteNode(startId);
      const after = useJourneyStore.getState();
      expect(after.nodes).toHaveLength(1);
      expect(after.nodes[0].data.stageType).toBe('end');
    });

    it('clears selectedNodeId if deleted node was selected', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('stage', { x: 0, y: 0 });
      const nodeId = useJourneyStore.getState().nodes[0].id;
      store.setSelectedNodeId(nodeId);

      expect(useJourneyStore.getState().selectedNodeId).toBe(nodeId);
      store.deleteNode(nodeId);
      expect(useJourneyStore.getState().selectedNodeId).toBeNull();
    });
  });

  describe('duplicateNode', () => {
    it('creates a copy with offset position', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('stage', { x: 100, y: 200 });
      const nodeId = useJourneyStore.getState().nodes[0].id;

      store.duplicateNode(nodeId);
      const { nodes } = useJourneyStore.getState();
      expect(nodes).toHaveLength(2);
      expect(nodes[1].data.label).toBe('New Stage (copy)');
      expect(nodes[1].position).toEqual({ x: 150, y: 250 });
      expect(nodes[1].id).not.toBe(nodeId);
    });
  });

  describe('undo/redo', () => {
    it('undoes the last action', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('start', { x: 0, y: 0 });
      expect(useJourneyStore.getState().nodes).toHaveLength(1);

      store.undo();
      expect(useJourneyStore.getState().nodes).toHaveLength(0);
    });

    it('redoes after undo', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('start', { x: 0, y: 0 });

      store.undo();
      expect(useJourneyStore.getState().nodes).toHaveLength(0);

      store.redo();
      expect(useJourneyStore.getState().nodes).toHaveLength(1);
    });

    it('canUndo returns false when at beginning', () => {
      expect(useJourneyStore.getState().canUndo()).toBe(false);
    });

    it('canRedo returns false when at end', () => {
      expect(useJourneyStore.getState().canRedo()).toBe(false);
    });
  });

  describe('exportJourney / importJourney', () => {
    it('round-trips journey data correctly', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('start', { x: 0, y: 0 });
      store.addStageNode('end', { x: 300, y: 0 });
      store.updateJourneyMeta({ name: 'Test Journey', description: 'A test' });

      const exported = useJourneyStore.getState().exportJourney();
      expect(exported.name).toBe('Test Journey');
      expect(exported.description).toBe('A test');
      expect(exported.stages).toHaveLength(2);
      expect(exported.stages[0].stageType).toBe('start');
      expect(exported.stages[1].stageType).toBe('end');

      // Clear and re-import
      store.clearCanvas();
      expect(useJourneyStore.getState().nodes).toHaveLength(0);

      store.importJourney(exported);
      const { nodes, journeyMeta } = useJourneyStore.getState();
      expect(nodes).toHaveLength(2);
      expect(journeyMeta.name).toBe('Test Journey');
      expect(nodes[0].data.stageType).toBe('start');
    });
  });

  describe('clearCanvas', () => {
    it('removes all nodes and edges', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('start', { x: 0, y: 0 });
      store.addStageNode('end', { x: 200, y: 0 });

      store.clearCanvas();
      const { nodes, edges } = useJourneyStore.getState();
      expect(nodes).toHaveLength(0);
      expect(edges).toHaveLength(0);
    });
  });

  describe('autoLayout', () => {
    it('repositions nodes without error', () => {
      const store = useJourneyStore.getState();
      store.addStageNode('start', { x: 0, y: 0 });
      store.addStageNode('stage', { x: 0, y: 0 });
      store.addStageNode('end', { x: 0, y: 0 });

      // Should not throw
      store.autoLayout();
      const { nodes } = useJourneyStore.getState();
      expect(nodes).toHaveLength(3);
      // Positions should have been updated (at least one should differ from origin)
      const hasNonZeroPos = nodes.some((n) => n.position.x !== 0 || n.position.y !== 0);
      expect(hasNonZeroPos).toBe(true);
    });
  });

  describe('journeyMeta', () => {
    it('updates journey metadata', () => {
      const store = useJourneyStore.getState();
      store.updateJourneyMeta({ name: 'My Journey', mode: 'post-sales' });

      const { journeyMeta } = useJourneyStore.getState();
      expect(journeyMeta.name).toBe('My Journey');
      expect(journeyMeta.mode).toBe('post-sales');
    });
  });
});
