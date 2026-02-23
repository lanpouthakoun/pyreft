export type StageType = 'start' | 'stage' | 'end';

export type DetectionType = 'visual' | 'text' | 'url' | 'custom';

export interface DetectionConfig {
  type: DetectionType;
  criteria: string;
}

export interface JourneyStage {
  id: string;
  name: string;
  instruction: string;
  detectionConfig: DetectionConfig;
  stageType: StageType;
}

export interface JourneyTransition {
  id: string;
  source: string;
  target: string;
  condition?: string;
}

export interface Journey {
  id: string;
  name: string;
  description: string;
  mode: 'pre-sales' | 'post-sales';
  stages: JourneyStage[];
  transitions: JourneyTransition[];
  version: number;
  status: 'draft' | 'published';
  createdAt: string;
  updatedAt: string;
}

export interface StageNodeData extends Record<string, unknown> {
  label: string;
  instruction: string;
  stageType: StageType;
  detectionConfig: DetectionConfig;
}
