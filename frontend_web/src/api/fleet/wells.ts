// Copyright 2026 DataRobot, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// STATIC UI METADATA from the agent spec's `wells` block. Not a model input,
// never evidence, never scored. Coordinates are ILLUSTRATIVE - four wells on
// one offshore block for a demo, not a real operator's positions.
export interface WellPosition {
  well: string;
  latitude: number;
  longitude: number;
}

export const WELL_POSITIONS: WellPosition[] = [
  { well: 'GUY-001', latitude: 7.42, longitude: -57.12 },
  { well: 'GUY-002', latitude: 7.35, longitude: -57.04 },
  { well: 'GUY-003', latitude: 7.48, longitude: -56.98 },
  { well: 'GUY-004', latitude: 7.29, longitude: -57.19 },
];

export const MAP_CENTRE: [number, number] = [7.385, -57.08];
export const MAP_ZOOM = 9;
export const MAP_LABEL = 'Offshore Guyana - illustrative block';

// Colour BY THE STATE WORD, never by pit rate or probability alone. A routine
// mud transfer must read routine even when its pit gains faster than a kick.
// Each colour is a fixed hue that reads in both light and dark themes.
export interface StateStyle {
  fill: string;
  stroke: string;
  label: string;
}

const NOT_SCORED: StateStyle = {
  // Neutral grey for the "not yet scored" first paint.
  fill: '#9ca3af',
  stroke: '#6b7280',
  label: 'not scored',
};

const STATE_STYLES: Record<string, StateStyle> = {
  // Formation events - alarm palette.
  kick: { fill: '#ef4444', stroke: '#b91c1c', label: 'kick' },
  'lost circulation': { fill: '#f97316', stroke: '#c2410c', label: 'lost circulation' },
  watch: { fill: '#eab308', stroke: '#a16207', label: 'watch' },
  // Routine - calm palette.
  quiet: { fill: '#22c55e', stroke: '#15803d', label: 'quiet' },
  'rate change': { fill: '#3b82f6', stroke: '#1d4ed8', label: 'rate change' },
  'mud transfer': { fill: '#06b6d4', stroke: '#0e7490', label: 'mud transfer' },
};

export function styleForState(state: string | null | undefined): StateStyle {
  if (!state) {
    return NOT_SCORED;
  }
  return STATE_STYLES[state] ?? NOT_SCORED;
}

export { NOT_SCORED };
